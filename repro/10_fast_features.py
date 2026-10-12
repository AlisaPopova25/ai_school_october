"""Дают ли «быстрые» признаки команды прирост: базовые признаки против базовых + быстрых (LOSO, сон/бодрствование).

    python repro/10_fast_features.py                         # CatBoost и нейросеть статьи
    python repro/10_fast_features.py --models CatBoost       # только CatBoost
    python repro/10_fast_features.py --smoke                 # проверка за минуту: 6 человек, крошечные модели
    (из корня репозитория; всё сразу — python repro/run_all.py)

Быстрые признаки (repro/fast_features.py): пик движения за эпоху f_peak, доля активных 5-секундных отрезков в
минутном окне f_active, относительный пульс за минуту f_hr. База — признаки статьи motion, hr и время ночи:
clock (как в статье) или cosine (как в эксперименте команды с MLP). Наборы признаков:
    Motion, HR, Clock            Motion, HR, Clock + быстрые
    Motion, HR, Cosine           Motion, HR, Cosine + быстрые
Модели — те же, что в repro/09_paper_metrics.py (CatBoost из catboost_config.py). Метрики: AUC, AP wake; при
«сон найден» = 0.9 — accuracy, wake найден, κ, F1 wake, macro F1; при пороге P(сон) ≥ 0.5 (как в эксперименте
команды) — accuracy, wake найден, κ, F1 wake.

Результат: outputs/repro/fast_features/results.csv, results.md, fig_fast_features.png; предсказания кэшируются в
outputs/repro/fast_features/oof/ (повторный запуск обучает только недостающее). Отчёт RESULTS.md подхватит их сам.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score  # noqa: E402

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(REPRO.parent))
import fast_features as ff  # noqa: E402
import sleeplib as sl  # noqa: E402
from progress import Progress, fmt_time  # noqa: E402

N_JOBS = int(os.environ.get("SLEEP_N_JOBS", 6))
OUT = sl.OUT / "fast_features"
BASE = {"Clock": ["motion", "hr", "clock"], "Cosine": ["motion", "hr", "cosine"]}
MODELS = ["CatBoost", "Neural Net"]


def out_dir(smoke):
    return sl.OUT / "smoke" / "fast_features" if smoke else OUT


def paper_metrics():
    """repro/09_paper_metrics.py как модуль (имя начинается с цифры, обычный import не работает)."""
    if "paper_metrics_09" not in sys.modules:
        spec = importlib.util.spec_from_file_location("paper_metrics_09", REPRO / "09_paper_metrics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["paper_metrics_09"] = module
        spec.loader.exec_module(module)
    return sys.modules["paper_metrics_09"]


def feature_sets(bases):
    sets = {}
    for base in bases:
        sets[f"Motion, HR, {base}"] = BASE[base]
        sets[f"Motion, HR, {base} + быстрые"] = BASE[base] + ff.FAST_FEATURES
    return sets


def _fold(data, model_name, features, test, subjects, smoke):
    """Один фолд LOSO (параметры явно: процесс-работник joblib импортирует модули заново)."""
    X, y, _ = sl.stack(data, [s for s in subjects if s != test], features)
    model = paper_metrics().make_model(model_name, "sw", smoke).fit(X, sl.sleep_wake(y))
    return test, model.predict_proba(sl.stack(data, [test], features)[0])[:, 1]


def loso(data, model_name, fs, features, smoke, bar, status):
    path = out_dir(smoke) / "oof" / f"{model_name.lower().replace(' ', '_')}__{fs}.npz"
    if path.exists():
        z = np.load(path)
        return {s: z[s] for s in sl.SUBJECTS}
    subjects = list(sl.SUBJECTS)
    jobs = (delayed(_fold)(data, model_name, features, s, subjects, smoke) for s in subjects)
    try:
        stream = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(jobs)
    except TypeError:
        stream = Parallel(n_jobs=N_JOBS)(jobs)
    result = {}
    for subject, p in stream:
        result[subject] = p
        bar.update(status=f"{status} | фолд {len(result)}/{len(subjects)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    return result


def metrics(y, p):
    """Метрики сон/бодрствование: AUC, AP wake, при 90% сна и при пороге 0.5."""
    row = dict(auc=roc_auc_score(y, p), ap_wake=average_precision_score(1 - y, 1 - p))
    at90 = next(r for r in paper_metrics().sleep_wake_metrics(y, p) if r["sensitivity_target"] == 0.9)
    row.update({f"{k}@90": at90[k] for k in ["accuracy", "wake_correct", "kappa", "f1_wake", "f1_macro"]})
    pred = (p >= 0.5).astype(int)
    row.update({"accuracy@0.5": np.mean(pred == y), "wake_correct@0.5": np.mean(pred[y == 0] == 0),
                "kappa@0.5": cohen_kappa_score(y, pred), "f1_wake@0.5": sl.sw_f1(y, pred)["f1_wake"]})
    return row


NAMES = {"auc": "AUC", "ap_wake": "AP wake", "accuracy@90": "accuracy @90% сна", "wake_correct@90": "wake найден @90%",
         "kappa@90": "κ @90%", "f1_wake@90": "F1 wake @90%", "f1_macro@90": "macro F1 @90%",
         "accuracy@0.5": "accuracy @0.5", "wake_correct@0.5": "wake найден @0.5", "kappa@0.5": "κ @0.5",
         "f1_wake@0.5": "F1 wake @0.5"}


def write_markdown(df, path):
    cols = ["model", "feature_set"] + list(NAMES)
    lines = ["| модель | признаки | " + " | ".join(NAMES.values()) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append(f"| {r.model} | {r.feature_set} | " + " | ".join(f"{r[c]:.3f}" for c in NAMES) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, col, title in zip(axes, ["auc", "kappa@90", "f1_wake@90"], ["AUC", "κ при 90% сна", "F1 wake при 90% сна"]):
        labels = [f"{m}\n{fs}" for m, fs in zip(df.model, df.feature_set)]
        colors = ["#0f766e" if "быстрые" in fs else "#9aa0b5" for fs in df.feature_set]
        ax.barh(range(len(df)), df[col], color=colors)
        for i, v in enumerate(df[col]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        ax.set_yticks(range(len(df)), labels if ax is axes[0] else [""] * len(df), fontsize=8)
        ax.invert_yaxis()
        lo = df[col].min()
        ax.set_xlim(lo - (df[col].max() - lo) * 0.6 - 0.005, df[col].max() + (df[col].max() - lo) * 0.4 + 0.005)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=.3)
    fig.suptitle("Базовые признаки (серые) против базовых + быстрых (зелёные), LOSO", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Базовые признаки против базовых + быстрых (LOSO)")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        choices=["Logistic Regression", "k-Nearest Neighbors", "Random Forest", "Neural Net", "CatBoost"])
    parser.add_argument("--base", nargs="+", default=["Clock", "Cosine"], choices=list(BASE),
                        help="чем задаётся время ночи в базовом наборе")
    parser.add_argument("--smoke", action="store_true", help="6 человек и крошечные модели — проверить, что работает")
    args = parser.parse_args()
    if args.smoke:
        sl.SUBJECTS[:] = sl.SUBJECTS[:6]
    m09 = paper_metrics()
    models = list(args.models)
    if "CatBoost" in models and not m09.catboost_available():
        print('CatBoost пропущен: пакет не установлен (pip install "catboost>=1.2" или python repro/run_all.py)')
        models.remove("CatBoost")
    started = time.time()
    data = ff.add_to(sl.load_paper_features())
    sets = feature_sets(args.base)
    y_all = np.concatenate([sl.sleep_wake(data[s]["stage"]) for s in sl.SUBJECTS])
    todo = [(m, fs) for m in models for fs in sets]
    bar = Progress(len(todo) * len(sl.SUBJECTS), "быстрые признаки")
    bar.message(f"{len(todo)} этапов LOSO по {len(sl.SUBJECTS)} фолдов: модели {', '.join(models)}; "
                "уже посчитанное берётся из сохранённого")
    rows = []
    for step, (model_name, fs) in enumerate(todo, 1):
        p = loso(data, model_name, fs, sets[fs], args.smoke, bar, f"этап {step}/{len(todo)}: {model_name} | {fs}")
        rows.append(dict(model=model_name, feature_set=fs,
                         **metrics(y_all, np.concatenate([p[s] for s in sl.SUBJECTS]))))
    bar.close()
    out = out_dir(args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False, float_format="%.4f")
    write_markdown(df, out / "results.md")
    figure(df, out / "fig_fast_features.png")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df[["model", "feature_set", "auc", "kappa@90", "f1_wake@90", "kappa@0.5"]].round(3).to_string(index=False))
    print(f"\nГотово за {fmt_time(time.time() - started)}: {out.relative_to(sl.ROOT)}/results.md")


if __name__ == "__main__":
    main()
