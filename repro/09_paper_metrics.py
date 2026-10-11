"""ВСЕ метрики статьи Walch et al. (2019) + F1 и точность ответа (PPV) — для всех четырёх моделей статьи.

    python repro/09_paper_metrics.py                         # все 4 модели (дольше всех — нейросеть)
    python repro/09_paper_metrics.py --models "Logistic Regression" "k-Nearest Neighbors" "Random Forest"
                                                             # без нейросети — несколько минут
    python repro/09_paper_metrics.py --models "Neural Net"   # дообучить только нейросеть; остальные — из сохранённых
    python repro/09_paper_metrics.py --plots-only            # ничего не обучать, перерисовать из сохранённых
    (из корня репозитория; в VS Code — конфигурация «9. All paper metrics»)

Обучаются модели из --models; в таблицы и на рисунки попадают ВСЕ модели, чьи предсказания уже сохранены
(outputs/repro/paper_metrics/oof/). Рисунки 5–7 в статье построены по нейросети; пока её нет, они строятся
по случайному лесу (это подписано в файлах и в отчёте) и сами переключатся на нейросеть, когда она появится.

Протокол: leave-one-subject-out (31 фолд) — каждый человек предсказывается моделью, обученной на остальных 30.
Модели и гиперпараметры — из кода статьи (source/utils.py), фиксированные значения из её сеток перебора
(полный перебор GridSearch для нейросети на каждом фолде занял бы часы):
  Logistic Regression  liblinear, L2, C = 0.001 (сон/бодрствование) / 0.1 (три класса) — что выбрал GridSearch статьи
                       на всех данных; веса классов balanced; три класса — one-vs-rest, как в sklearn 0.20
  k-Nearest Neighbors  k = 500, weights = distance
  Random Forest        100 деревьев, max_depth 10, min_samples_leaf 32, max_features 1.0, веса классов balanced
  Neural Net           MLP (15, 15, 15), relu, adam, alpha 0.01, max_iter 2000, n_iter_no_change 20
Признаки — как в статье: Motion only / HR only / Motion, HR / Motion, HR, Clock.

Что считается (outputs/repro/paper_metrics/):
  table1_*            аналог Таблицы 1: TST, TIB, SOL, WASO, SE, REM, NREM по PSG, рядом — значения статьи
  sleep_wake_*.csv    Таблицы 2–5: при «сон найден» 0.80/0.90/0.93/0.95 — accuracy, wake найден (specificity),
                      сон найден (sensitivity), κ, AUC; плюс AP (площадь под PR-кривой wake), PPV wake/сон,
                      F1 wake, F1 сон, macro F1
  three_class.csv     Таблица 6: wake найден 0.6, NREM/REM найдены (уравнены), лучшая accuracy и κ; плюс macro F1
                      и F1 по стадиям
  bland_altman*.csv   Рис. 5: TST, SOL, WASO, SE, REM, NREM — сдвиг, SD, границы согласия (нейросеть, θW = 0.3, θREM = 0.35)
  per_subject*.csv    Рис. 6–7: качество по каждому человеку
  fig2–fig7 *.png     аналоги рисунков 2–7 статьи
Рисунки/таблицы MESA (табл. 7–9, рис. 8–9) не воспроизводимы: данные MESA доступны только по заявке NSRR.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_curve, roc_auc_score, \
    roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sleeplib as sl  # noqa: E402
from source.utils import LiblinearLogisticRegression  # noqa: E402

warnings.filterwarnings("ignore")
OUT = sl.OUT / "paper_metrics"
CACHE = OUT / "oof"
N_JOBS = int(os.environ.get("SLEEP_N_JOBS", 6))
SENSITIVITIES = (0.80, 0.90, 0.93, 0.95)
THETA_WAKE, THETA_REM = 0.3, 0.35  # фиксированные пороги статьи (рис. 5, 6)

CLASSIFIERS = ["Logistic Regression", "k-Nearest Neighbors", "Random Forest", "Neural Net"]
FEATURE_SETS = {"Motion only": ["motion"], "HR only": ["hr"], "Motion, HR": ["motion", "hr"],
                "Motion, HR, Clock": ["motion", "hr", "clock"]}
COLORS = {"Motion only": "#3b5b92", "HR only": "#f0a202", "Motion, HR": "#39ad48", "Motion, HR, Clock": "#d3494e"}
PAPER_TABLE1 = {  # Таблица 1 статьи: среднее (SD)
    "TST, мин": (427.87, 38.87), "TIB, мин": (472.56, 27.03), "SOL, мин": (14.97, 10.1), "WASO, мин": (28.73, 22.8),
    "SE, %": (90.48, 5.54), "REM, мин": (107.15, 31.22), "NREM, мин": (320.77, 39.11)}


def make_model(name, target):
    if name == "Logistic Regression":
        return LiblinearLogisticRegression(penalty="l2", solver="liblinear", C=0.001 if target == "sw" else 0.1,
                                           class_weight="balanced")
    if name == "k-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=500, weights="distance")
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_features=1.0, max_depth=10, min_samples_split=10,
                                      min_samples_leaf=32, class_weight="balanced", n_jobs=1, random_state=0)
    return MLPClassifier(activation="relu", hidden_layer_sizes=(15, 15, 15), alpha=0.01, solver="adam",
                         max_iter=2000, n_iter_no_change=20, random_state=0)


def slug(text):
    return text.lower().replace(", ", "_").replace(" ", "_").replace("-", "_")


# --------------------------------------------------------------------------- обучение (LOSO) с кэшем
def loso_predictions(data, clf, fs, target):
    path = CACHE / f"{slug(clf)}__{slug(fs)}__{target}.npz"
    if path.exists():
        z = np.load(path)
        return {s: z[s] for s in sl.SUBJECTS}
    conv = sl.sleep_wake if target == "sw" else sl.three_class
    features = FEATURE_SETS[fs]

    def fold(test):
        X, y, _ = sl.stack(data, [s for s in sl.SUBJECTS if s != test], features)
        model = make_model(clf, target).fit(X, conv(y))
        return test, model.predict_proba(sl.stack(data, [test], features)[0])

    result = dict(Parallel(n_jobs=N_JOBS)(delayed(fold)(s) for s in sl.SUBJECTS))
    np.savez(path, **result)
    return result


# --------------------------------------------------------------------------- метрики
def sleep_wake_metrics(y, p_sleep):
    """Таблицы 2–5 + AP, PPV, F1 на всех эпохах (out-of-fold)."""
    rows = []
    auc = roc_auc_score(y, p_sleep)
    ap = average_precision_score(1 - y, 1 - p_sleep)  # wake как положительный класс (рис. 3)
    for target in SENSITIVITIES:
        threshold = np.quantile(p_sleep[y == 1], 1 - target)
        pred = (p_sleep >= threshold).astype(int)
        f1 = sl.sw_f1(y, pred)
        rows.append(dict(sensitivity_target=target, accuracy=np.mean(pred == y), wake_correct=np.mean(pred[y == 0] == 0),
                         sleep_correct=np.mean(pred[y == 1] == 1), kappa=cohen_kappa_score(y, pred), auc=auc,
                         ap_wake=ap, ppv_wake=np.mean(y[pred == 0] == 0) if np.any(pred == 0) else np.nan,
                         ppv_sleep=np.mean(y[pred == 1] == 1), **f1))
    return rows


def table1(data):
    """Аналог Таблицы 1 статьи по разметке PSG (все размеченные эпохи, если есть data/labels)."""
    rows = []
    for s in sl.SUBJECTS:
        path = sl.ROOT / "data" / "labels" / f"{s}_labeled_sleep.txt"
        stages = np.loadtxt(path)[:, 1] if path.exists() else data[s]["stage"]
        stages = stages[stages >= 0]
        m = sl.sleep_metrics(sl.three_class(stages))
        rows.append({"subject": s, "TST, мин": m["TST"], "TIB, мин": len(stages) * 0.5, "SOL, мин": m["SOL"],
                     "WASO, мин": m["WASO"], "SE, %": 100 * m["SE"], "REM, мин": m["REM"], "NREM, мин": m["NREM"]})
    per = pd.DataFrame(rows)
    summary = []
    for col, (pm, psd) in PAPER_TABLE1.items():
        v = per[col]
        summary.append({"показатель": col, "наши: среднее": v.mean(), "наши: SD": v.std(), "наши: мин": v.min(),
                        "наши: макс": v.max(), "статья: среднее": pm, "статья: SD": psd})
    source = "data/labels" if (sl.ROOT / "data" / "labels").exists() else "outputs/features (валидные эпохи)"
    return per, pd.DataFrame(summary), source


def bland_altman(probas3, y3s):
    """Рис. 5: показатели сна PSG и модели для каждого человека при фиксированных порогах статьи."""
    rows = []
    for s in sl.SUBJECTS:
        pred = sl.fixed_threshold_predict(probas3[s], THETA_WAKE, THETA_REM)
        true_m, pred_m = sl.sleep_metrics(y3s[s]), sl.sleep_metrics(pred)
        for key in ["TST", "SOL", "WASO", "SE", "REM", "NREM"]:
            rows.append(dict(subject=s, metric=key, psg=true_m[key], model=pred_m[key], diff=true_m[key] - pred_m[key]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- рисунки
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25)


def fig_curves(preds_sw, y_all, kind, models):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")  # модели, которых ещё нет (например, нейросеть не обучена)
    for ax, clf in zip(axes.flat, models):
        for fs in FEATURE_SETS:
            p = preds_sw[(clf, fs)]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y_all, p)
                ax.plot(fpr, tpr, color=COLORS[fs], label=f"{fs} (AUC {roc_auc_score(y_all, p):.3f})")
            else:
                precision, recall, _ = precision_recall_curve(1 - y_all, 1 - p)
                ax.plot(recall, precision, color=COLORS[fs],
                        label=f"{fs} (AP {average_precision_score(1 - y_all, 1 - p):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ":", color="#999")
            ax.set_xlabel("доля wake, названная сном")
            ax.set_ylabel("доля сна, названная сном")
        else:
            ax.axhline(np.mean(y_all == 0), ls=":", color="#999")
            ax.set_xlabel("доля wake, названная wake (recall)")
            ax.set_ylabel("доля верных среди названных wake (precision)")
        ax.set_title(clf, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right" if kind == "roc" else "upper right")
        style(ax)
    fig.suptitle("Рис. 2 (аналог): ROC сон/бодрствование, LOSO" if kind == "roc" else
                 "Рис. 3 (аналог): precision–recall для wake, LOSO", fontsize=14)
    fig.tight_layout()
    name = "fig2_roc_sleep_wake.png" if kind == "roc" else "fig3_pr_wake.png"
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def fig4_three_class(curves, models):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    for ax, clf in zip(axes.flat, models):
        for fs in FEATURE_SETS:
            c = curves[(clf, fs)]
            ax.plot(c.wake_fpr, (c.nrem_correct + c.rem_correct) / 2, color=COLORS[fs], label=fs)
            if fs == "Motion only":  # как в статье: для одной motion показываем NREM и REM отдельно
                ax.plot(c.wake_fpr, c.nrem_correct, ":", color=COLORS[fs], label="Motion only: NREM")
                ax.plot(c.wake_fpr, c.rem_correct, "--", color=COLORS[fs], label="Motion only: REM")
        ax.set_title(clf, fontweight="bold")
        ax.set_xlabel("доля wake, названная REM или NREM")
        ax.set_ylabel("доля верных NREM и REM (уравненных)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
        style(ax)
    fig.suptitle("Рис. 4 (аналог): wake / NREM / REM, LOSO", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_three_class.png", dpi=130)
    plt.close(fig)


def fig5_bland_altman(ba_all, ba_stats, model_name):
    titles = {"TST": "TST, мин", "SOL": "SOL, мин", "WASO": "WASO, мин", "SE": "SE (доля)", "REM": "REM, мин",
              "NREM": "NREM, мин"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for ax, metric in zip(axes.flat, titles):
        for fs in FEATURE_SETS:
            d = ba_all[(ba_all.feature_set == fs) & (ba_all.metric == metric)]
            ax.scatter(d.psg, d["diff"], s=18 if fs == "Motion, HR, Clock" else 10, color=COLORS[fs], alpha=.85,
                       label=fs)
        st = ba_stats[(ba_stats.feature_set == "Motion, HR, Clock") & (ba_stats.metric == metric)].iloc[0]
        ax.axhline(0, color="#999", lw=.8)
        ax.axhline(st.bias, color=COLORS["Motion, HR, Clock"], lw=1.2)
        for lim in (st.loa_low, st.loa_high):
            ax.axhline(lim, color=COLORS["Motion, HR, Clock"], lw=1, ls="--")
        ax.set_title(f"{titles[metric]}: сдвиг {st.bias:.2f}, границы [{st.loa_low:.2f}; {st.loa_high:.2f}]",
                     fontsize=10)
        ax.set_xlabel(f"PSG, {titles[metric]}")
        ax.set_ylabel("PSG − модель")
        style(ax)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Рис. 5 (аналог): Bland–Altman, {model_name}, θW = 0.3, θREM = 0.35; линии — сдвиг и ±1.96 SD "
                 "для Motion, HR, Clock", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_bland_altman.png", dpi=130)
    plt.close(fig)


def fig6_7_per_subject(per6, per7, model_name):
    cols = [("accuracy", "accuracy"), ("wake_correct", "wake найден"), ("sleep_correct", "сон найден"), ("kappa", "κ"),
            ("f1_wake", "F1 wake")]
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 3.6))
    for ax, (col, title) in zip(axes, cols):
        ax.hist(per6[col].dropna(), bins=np.linspace(0, 1, 21), color=COLORS["Motion, HR, Clock"], edgecolor="white")
        ax.axvline(per6[col].median(), color="k", ls="--", lw=1)
        ax.set_title(f"{title}: медиана {per6[col].median():.2f}", fontsize=10)
        ax.set_xlabel("по людям")
        style(ax)
    axes[0].set_ylabel("людей")
    fig.suptitle(f"Рис. 6 (аналог): качество по людям, {model_name}, Motion, HR, Clock, P(wake) ≥ 0.3 → wake",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_per_subject_theta.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(SENSITIVITIES), 2, figsize=(10, 11), sharex=True)
    for i, tpr in enumerate(SENSITIVITIES):
        d = per7[per7.tpr == tpr]
        for j, (col, title) in enumerate([("accuracy", "accuracy"), ("wake_correct", "wake найден (specificity)")]):
            ax = axes[i, j]
            ax.hist(d[col].dropna(), bins=np.linspace(0, 1, 21), color="#2a7ab9", edgecolor="white")
            ax.axvline(d[col].median(), color="k", ls="--", lw=1)
            ax.set_title(f"сон найден = {tpr:.2f}: {title}, медиана {d[col].median():.2f}", fontsize=9)
            style(ax)
    fig.suptitle(f"Рис. 7 (аналог), {model_name}: порог подобран для каждого человека под заданную долю сна",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_per_subject_tpr.png", dpi=130)
    plt.close(fig)


def fig_summary(sw, models):
    """Сводка: AUC и F1 wake для всех моделей и наборов признаков (при 90% сна)."""
    d = sw[sw.sensitivity_target == 0.9]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=False)
    width = 0.2
    x = np.arange(len(models))
    for ax, (col, title) in zip(axes, [("auc", "AUC"), ("kappa", "κ при 90% сна"), ("f1_wake", "F1 wake при 90% сна")]):
        for i, fs in enumerate(FEATURE_SETS):
            vals = [d[(d.classifier == c) & (d.feature_set == fs)][col].iloc[0] for c in models]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, color=COLORS[fs], label=fs)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(x, [c.replace(" ", "\n", 1) for c in models], fontsize=8)
        ax.set_title(title, fontweight="bold")
        style(ax)
    axes[0].set_ylim(0.5, 0.95)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Все модели статьи и наборы признаков, LOSO", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_summary_models.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def cached(clf):
    return all((CACHE / f"{slug(clf)}__{slug(fs)}__{t}.npz").exists() for fs in FEATURE_SETS for t in ("sw", "3"))


def main():
    parser = argparse.ArgumentParser(description="Все метрики статьи + F1 для моделей статьи (LOSO)")
    parser.add_argument("--models", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS,
                        help="какие модели обучать (по умолчанию все четыре); в отчёт идут все уже посчитанные")
    parser.add_argument("--plots-only", action="store_true",
                        help="ничего не обучать: таблицы и рисунки из сохранённых предсказаний")
    parser.add_argument("--no-report", action="store_true",
                        help="не собирать outputs/RESULTS.md в конце (run_all.py собирает его сам)")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    data = sl.load_paper_features()
    y_sw = {s: sl.sleep_wake(data[s]["stage"]) for s in sl.SUBJECTS}
    y_3 = {s: sl.three_class(data[s]["stage"]) for s in sl.SUBJECTS}
    Y_sw = np.concatenate([y_sw[s] for s in sl.SUBJECTS])
    Y_3 = np.concatenate([y_3[s] for s in sl.SUBJECTS])

    per_t1, sum_t1, t1_source = table1(data)
    per_t1.to_csv(OUT / "table1_per_subject.csv", index=False, float_format="%.2f")
    sum_t1.to_csv(OUT / "table1_summary.csv", index=False, float_format="%.2f")
    (OUT / "table1_source.txt").write_text(t1_source, encoding="utf-8")

    # 1. обучение (с кэшем): только выбранные модели
    if not args.plots_only:
        for clf in args.models:
            for fs in FEATURE_SETS:
                print(f"обучение: {clf} / {fs}", flush=True)
                loso_predictions(data, clf, fs, "sw")
                loso_predictions(data, clf, fs, "3")

    # 2. метрики и рисунки: все модели, чьи предсказания сохранены полностью
    models = [c for c in CLASSIFIERS if cached(c)]
    if not models:
        sys.exit("Нет сохранённых предсказаний: запустите без --plots-only")
    per_subject_model = "Neural Net" if "Neural Net" in models else (
        "Random Forest" if "Random Forest" in models else models[-1])
    print("в отчёте модели:", ", ".join(models), "| рис. 5–7 по модели:", per_subject_model, flush=True)
    (OUT / "models_included.txt").write_text("\n".join(models), encoding="utf-8")
    (OUT / "per_subject_model.txt").write_text(per_subject_model, encoding="utf-8")

    sw_rows, tc_rows, curves, preds_sw, ba_frames = [], [], {}, {}, []
    for clf in models:
        for fs in FEATURE_SETS:
            p_sw = loso_predictions(data, clf, fs, "sw")
            p_3 = loso_predictions(data, clf, fs, "3")
            P_sw = np.concatenate([p_sw[s][:, 1] for s in sl.SUBJECTS])
            P_3 = np.vstack([p_3[s] for s in sl.SUBJECTS])
            preds_sw[(clf, fs)] = P_sw
            for r in sleep_wake_metrics(Y_sw, P_sw):
                sw_rows.append(dict(classifier=clf, feature_set=fs, **r))
            r6 = sl.three_class_at_wake_fpr(Y_3, P_3, 0.4)
            per_class = sl.three_class_f1(Y_3, sl.fixed_threshold_predict(P_3, THETA_WAKE, THETA_REM))
            tc_rows.append(dict(classifier=clf, feature_set=fs, wake_correct=r6["wake_correct"],
                                nrem_correct=r6["nrem_correct"], rem_correct=r6["rem_correct"],
                                best_accuracy=r6["best_accuracy"], kappa=r6["best_kappa"],
                                f1_macro_best=r6["best_f1_macro"], f1_macro_balanced=r6["balanced_f1_macro"],
                                f1_rem_balanced=r6["balanced_f1_rem"],
                                f1_wake_fixed=per_class["f1_wake"], f1_nrem_fixed=per_class["f1_nrem"],
                                f1_rem_fixed=per_class["f1_rem"], f1_macro_fixed=per_class["f1_macro"]))
            curves[(clf, fs)] = pd.DataFrame([sl.three_class_at_wake_fpr(Y_3, P_3, f)
                                              for f in np.linspace(0.05, 0.95, 19)])
            if clf == per_subject_model:
                ba = bland_altman(p_3, y_3)
                ba["feature_set"] = fs
                ba_frames.append(ba)
                if fs == "Motion, HR, Clock":
                    best_sw = p_sw

    sw = pd.DataFrame(sw_rows)
    tc = pd.DataFrame(tc_rows)
    sw.to_csv(OUT / "sleep_wake_all_models.csv", index=False, float_format="%.4f")
    tc.to_csv(OUT / "three_class_all_models.csv", index=False, float_format="%.4f")

    ba_all = pd.concat(ba_frames)
    ba_all.to_csv(OUT / "bland_altman_per_subject.csv", index=False, float_format="%.4f")
    ba_stats = ba_all.groupby(["feature_set", "metric"], sort=False)["diff"].agg(["mean", "std"]).reset_index()
    ba_stats.columns = ["feature_set", "metric", "bias", "sd"]
    ba_stats["loa_low"] = ba_stats.bias - 1.96 * ba_stats.sd
    ba_stats["loa_high"] = ba_stats.bias + 1.96 * ba_stats.sd
    ba_stats.to_csv(OUT / "bland_altman_stats.csv", index=False, float_format="%.4f")

    per6, per7 = [], []
    for s in sl.SUBJECTS:
        y, p_wake = y_sw[s], best_sw[s][:, 0]
        pred = np.where(p_wake >= THETA_WAKE, 0, 1)
        per6.append(dict(subject=s, accuracy=np.mean(pred == y),
                         wake_correct=np.mean(pred[y == 0] == 0) if np.any(y == 0) else np.nan,
                         sleep_correct=np.mean(pred[y == 1] == 1), kappa=cohen_kappa_score(y, pred),
                         f1_wake=sl.sw_f1(y, pred)["f1_wake"] if np.any(y == 0) else np.nan))
        for tpr in SENSITIVITIES:
            threshold = np.quantile(best_sw[s][y == 1, 1], 1 - tpr)
            pred = (best_sw[s][:, 1] >= threshold).astype(int)
            per7.append(dict(subject=s, tpr=tpr, accuracy=np.mean(pred == y),
                             wake_correct=np.mean(pred[y == 0] == 0) if np.any(y == 0) else np.nan))
    per6, per7 = pd.DataFrame(per6), pd.DataFrame(per7)
    per6.to_csv(OUT / "per_subject_theta.csv", index=False, float_format="%.4f")
    per7.to_csv(OUT / "per_subject_tpr.csv", index=False, float_format="%.4f")

    fig_curves(preds_sw, Y_sw, "roc", models)
    fig_curves(preds_sw, Y_sw, "pr", models)
    fig4_three_class(curves, models)
    fig5_bland_altman(ba_all, ba_stats, per_subject_model)
    fig6_7_per_subject(per6, per7, per_subject_model)
    fig_summary(sw, models)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(sw[sw.sensitivity_target == 0.9][["classifier", "feature_set", "accuracy", "wake_correct", "kappa", "auc",
                                              "f1_wake", "f1_macro"]].round(3).to_string(index=False))
    print(tc[["classifier", "feature_set", "nrem_correct", "rem_correct", "best_accuracy", "kappa",
              "f1_macro_best"]].round(3).to_string(index=False))
    print(ba_stats[ba_stats.feature_set == "Motion, HR, Clock"].round(2).to_string(index=False))
    if "Neural Net" not in models:
        print('\nНейросети пока нет. Обучить её (остальное возьмётся из сохранённого): '
              'python repro/09_paper_metrics.py --models "Neural Net"')
    if not args.no_report:  # все таблицы и рисунки — в один Markdown-файл outputs/RESULTS.md
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("08_results_markdown.py"))], check=True)


if __name__ == "__main__":
    main()
