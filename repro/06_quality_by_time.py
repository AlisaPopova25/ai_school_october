"""Гипотеза: качество предсказаний зависит от времени внутри ночи.

LOSO (31 фолд), признаки статьи. Для каждой эпохи — out-of-fold вероятности; затем метрики по часам от начала
ночи. Чтобы отделить «модель хуже» от «другой состав классов», считаем recall по каждому классу и AUC внутри часа,
а не только accuracy. Сравниваем с моделью без clock — clock кодирует время, и зависимость может быть от него.

    PYTHONPATH=. .venv/Scripts/python repro/06_quality_by_time.py
Результаты: outputs/repro/time_of_night/
"""
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleeplib as sl  # noqa: E402

warnings.filterwarnings("ignore")
OUT = sl.OUT / "time_of_night"
OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = int(os.environ.get("SLEEP_N_JOBS", 8))


def make(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              class_weight="balanced", random_state=0, verbose=-1, n_jobs=1)


def loso(data, features, kind, target):
    conv = sl.sleep_wake if target == "sw" else sl.three_class

    def fold(test):
        X, y, _ = sl.stack(data, [s for s in sl.SUBJECTS if s != test], features)
        return test, make(kind).fit(X, conv(y)).predict_proba(sl.stack(data, [test], features)[0])

    res = dict(Parallel(n_jobs=N_JOBS)(delayed(fold)(s) for s in sl.SUBJECTS))
    return np.vstack([res[s] for s in sl.SUBJECTS])


def main():
    data = sl.load_paper_features()
    stage = np.concatenate([data[s]["stage"] for s in sl.SUBJECTS])
    hours = np.concatenate([np.arange(len(data[s]["stage"])) * 0.5 / 60 for s in sl.SUBJECTS])
    subj = np.concatenate([[s] * len(data[s]["stage"]) for s in sl.SUBJECTS])
    hbin = np.minimum(hours.astype(int), 7)  # 0..7, последний бин = 7+ ч
    y_sw, y3 = sl.sleep_wake(stage), sl.three_class(stage)

    rows = []
    for fs_name, feats in [("motion,hr,clock", ["motion", "hr", "clock"]), ("motion,hr (без clock)", ["motion", "hr"])]:
        for kind in ["lr", "lgbm"]:
            p_sw = loso(data, feats, kind, "sw")[:, 1]
            p3 = loso(data, feats, kind, "3")
            # единый порог на всю ночь: sensitivity сна = 0.90 по всем эпохам
            thr = np.quantile(p_sw[y_sw == 1], 0.10)
            pred_sw = (p_sw >= thr).astype(int)
            pred3 = p3.argmax(1)  # модель с class_weight='balanced' → argmax сбалансирован
            for h in range(8):
                m = hbin == h
                row = dict(features=fs_name, model=kind, hour=f"{h}–{h + 1}" if h < 7 else "7+",
                           n=int(m.sum()), subjects=len(set(subj[m])),
                           share_wake=1 - y_sw[m].mean(), share_rem=np.mean(y3[m] == 2),
                           auc_sw=roc_auc_score(y_sw[m], p_sw[m]) if 0 < y_sw[m].mean() < 1 else np.nan,
                           wake_recall=np.mean(pred_sw[m][y_sw[m] == 0] == 0) if np.any(y_sw[m] == 0) else np.nan,
                           sleep_recall=np.mean(pred_sw[m][y_sw[m] == 1] == 1),
                           accuracy_sw=np.mean(pred_sw[m] == y_sw[m]),
                           kappa_sw=cohen_kappa_score(y_sw[m], pred_sw[m]) if 0 < y_sw[m].mean() < 1 else np.nan,
                           f1_wake_sw=sl.sw_f1(y_sw[m], pred_sw[m])["f1_wake"] if np.any(y_sw[m] == 0) else np.nan,
                           accuracy3=np.mean(pred3[m] == y3[m]),
                           f1_macro3=sl.three_class_f1(y3[m], pred3[m])["f1_macro"])
                for c, name in enumerate(["wake", "nrem", "rem"]):
                    row[f"recall3_{name}"] = np.mean(pred3[m][y3[m] == c] == c) if np.any(y3[m] == c) else np.nan
                rows.append(row)
            print(f"done {fs_name} {kind}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "quality_by_hour.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    hrs = df.hour.unique()
    for (fs, kind), g in df.groupby(["features", "model"], sort=False):
        ls = "-" if "clock" in fs and "без" not in fs else "--"
        col = "#2a7ab9" if kind == "lgbm" else "#c44e52"
        lab = f"{kind.upper()}: {fs}"
        axes[0].plot(hrs, g.auc_sw, ls, color=col, marker="o", ms=3, label=lab)
        axes[1].plot(hrs, g.wake_recall, ls, color=col, marker="o", ms=3, label=lab)
        if kind == "lgbm":
            axes[2].plot(hrs, g.recall3_nrem, ls, color="#4292c6", marker="o", ms=3, label=f"NREM ({fs})")
            axes[2].plot(hrs, g.recall3_rem, ls, color="#2ca02c", marker="o", ms=3, label=f"REM ({fs})")
    base = df[(df.model == "lgbm") & (df.features == "motion,hr,clock")]
    ax2 = axes[0].twinx()
    ax2.bar(hrs, base.share_wake, alpha=.15, color="#d62728")
    ax2.set_ylabel("доля wake в часе", color="#d62728")
    axes[0].set_title("AUC sleep/wake внутри часа")
    axes[1].set_title("Доля распознанного wake\n(единый порог: 90% сна за ночь)")
    axes[2].set_title("Recall NREM и REM по часам (LightGBM, 3 класса)")
    for ax in axes:
        ax.set_xlabel("часы от начала ночи")
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "quality_by_hour.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
