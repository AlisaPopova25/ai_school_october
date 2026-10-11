"""F1-мера для моделей проекта — рядом со старыми метриками (accuracy, доля верного wake, κ, AUC), не вместо них.

    PYTHONPATH=. python repro/07_f1_metrics.py
Результаты: outputs/repro/f1/

Что считается (leave-one-subject-out, 31 фолд, признаки статьи и расширенные):
  * сон/бодрствование: F1(wake) — wake как положительный (редкий, 8%) класс, F1(sleep), macro F1 (их среднее);
    при пороге «90% сна распознано» (как в Таблицах 2–5) и лучший F1(wake) по всем порогам;
  * три класса: macro F1 и F1 для wake / NREM / REM;
  * базовая «модель», которая всегда отвечает «сон» (или «NREM») — на ней видно, почему accuracy обманывает,
    а F1 — нет;
  * F1(wake) по людям — насколько метрика скачет от человека к человеку.
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_features as ef  # noqa: E402
import sleeplib as sl  # noqa: E402

warnings.filterwarnings("ignore")
OUT = sl.OUT / "f1"
OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = int(os.environ.get("SLEEP_N_JOBS", 6))
SENSITIVITY = 0.90  # порог как в Таблицах 2–5: модель находит 90% сна


def make(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, class_weight="balanced",
                              random_state=0, verbose=-1, n_jobs=1)


def loso(X, y, kind):
    """Out-of-fold вероятности: для каждого человека — модель, обученная на остальных 30."""
    def fold(test):
        train = [s for s in sl.SUBJECTS if s != test]
        model = make(kind).fit(np.vstack([X[s] for s in train]), np.concatenate([y[s] for s in train]))
        return test, model.predict_proba(X[test])

    result = dict(Parallel(n_jobs=N_JOBS)(delayed(fold)(s) for s in sl.SUBJECTS))
    return {s: result[s] for s in sl.SUBJECTS}


def sleep_wake_row(name, y, p_sleep):
    if p_sleep is None:  # «всегда сон»
        pred = np.ones_like(y)
        return dict(model=name, auc=0.5, accuracy=accuracy_score(y, pred),
                    wake_correct=0.0, kappa=0.0, **sl.sw_f1(y, pred), best_f1_wake=0.0)
    threshold = np.quantile(p_sleep[y == 1], 1 - SENSITIVITY)
    pred = (p_sleep >= threshold).astype(int)
    return dict(model=name, auc=roc_auc_score(y, p_sleep), accuracy=accuracy_score(y, pred),
                wake_correct=np.mean(pred[y == 0] == 0), kappa=cohen_kappa_score(y, pred),
                **sl.sw_f1(y, pred), **sl.best_f1_wake(y, p_sleep))


def three_class_row(name, y3, proba):
    pred = np.ones_like(y3) if proba is None else proba.argmax(1)  # «всегда NREM» / argmax сбалансированной модели
    return dict(model=name, accuracy=accuracy_score(y3, pred), kappa=cohen_kappa_score(y3, pred),
                **sl.three_class_f1(y3, pred))


def main():
    paper = sl.load_paper_features()
    ext, names = ef.load_all(paper)
    paper_cols = [names.index(n) for n in ("paper_motion", "paper_hr", "paper_clock")]
    stage = {s: paper[s]["stage"] for s in sl.SUBJECTS}
    y_sw = {s: sl.sleep_wake(stage[s]) for s in sl.SUBJECTS}
    y_3 = {s: sl.three_class(stage[s]) for s in sl.SUBJECTS}
    Y_sw = np.concatenate([y_sw[s] for s in sl.SUBJECTS])
    Y_3 = np.concatenate([y_3[s] for s in sl.SUBJECTS])

    experiments = [
        ("LR, признаки статьи", {s: ext[s][:, paper_cols] for s in sl.SUBJECTS}, "lr"),
        ("LightGBM, признаки статьи", {s: ext[s][:, paper_cols] for s in sl.SUBJECTS}, "lgbm"),
        ("LR, расширенные признаки", ext, "lr"),
        ("LightGBM, расширенные признаки", ext, "lgbm"),
    ]
    sw_rows = [sleep_wake_row("всегда «сон»", Y_sw, None)]
    tc_rows = [three_class_row("всегда «NREM»", Y_3, None)]
    per_subject, curves = [], {}
    for name, X, kind in experiments:
        print("LOSO:", name, flush=True)
        p_sw = loso(X, y_sw, kind)
        p_3 = loso(X, y_3, kind)
        P_sw = np.concatenate([p_sw[s][:, 1] for s in sl.SUBJECTS])
        sw_rows.append(sleep_wake_row(name, Y_sw, P_sw))
        tc_rows.append(three_class_row(name, Y_3, np.vstack([p_3[s] for s in sl.SUBJECTS])))
        curves[name] = P_sw
        threshold = np.quantile(P_sw[Y_sw == 1], 1 - SENSITIVITY)  # один порог на всех, как для общей таблицы
        for s in sl.SUBJECTS:
            pred = (p_sw[s][:, 1] >= threshold).astype(int)
            if np.any(y_sw[s] == 0):
                per_subject.append(dict(model=name, subject=s, wake_share=1 - y_sw[s].mean(),
                                        f1_wake=f1_score(y_sw[s], pred, pos_label=0, zero_division=0),
                                        accuracy=accuracy_score(y_sw[s], pred)))

    sw = pd.DataFrame(sw_rows)
    tc = pd.DataFrame(tc_rows)
    ps = pd.DataFrame(per_subject)
    sw.to_csv(OUT / "f1_sleep_wake.csv", index=False, float_format="%.4f")
    tc.to_csv(OUT / "f1_three_class.csv", index=False, float_format="%.4f")
    ps.to_csv(OUT / "f1_per_subject.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\nСон/бодрствование (порог: 90% сна распознано):")
    print(sw[["model", "accuracy", "wake_correct", "kappa", "auc", "f1_wake", "f1_sleep", "f1_macro",
              "best_f1_wake"]].round(3).to_string(index=False))
    print("\nТри класса (argmax):")
    print(tc.round(3).to_string(index=False))

    # ---------------- рисунок 1: accuracy против F1 — главная иллюстрация
    fig, ax = plt.subplots(figsize=(12, 4.8))
    metrics = [("accuracy", "accuracy", "#9e9e9e"), ("f1_sleep", "F1 сна", "#6baed6"),
               ("f1_wake", "F1 бодрствования", "#d62728"), ("f1_macro", "macro F1", "#2a7ab9")]
    width = 0.2
    x = np.arange(len(sw))
    for i, (col, label, color) in enumerate(metrics):
        bars = ax.bar(x + (i - 1.5) * width, sw[col], width, label=label, color=color)
        for b, v in zip(bars, sw[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, sw.model, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title("Accuracy почти не отличает модели от «всегда сон»; F1 бодрствования — отличает (LOSO, 90% сна)")
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "f1_vs_accuracy.png", dpi=130)
    plt.close(fig)

    # ---------------- рисунок 2: precision / recall / F1 бодрствования в зависимости от порога
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, name in zip(axes, ["LR, признаки статьи", "LR, расширенные признаки"]):
        y_wake = 1 - Y_sw
        p_wake = 1 - curves[name]
        precision, recall, thresholds = precision_recall_curve(y_wake, p_wake)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        ax.plot(thresholds, precision[:-1], color="#6baed6", label="precision (точность wake)")
        ax.plot(thresholds, recall[:-1], color="#9e9e9e", label="recall (полнота wake)")
        ax.plot(thresholds, f1[:-1], color="#d62728", lw=2, label="F1 wake")
        i = int(np.argmax(f1[:-1]))
        ax.scatter([thresholds[i]], [f1[i]], color="#d62728", zorder=3)
        ax.annotate(f"max F1 = {f1[i]:.2f}\nпри P(wake) ≥ {thresholds[i]:.2f}", (thresholds[i], f1[i]),
                    xytext=(10, 10), textcoords="offset points", fontsize=8)
        t90 = 1 - np.quantile(curves[name][Y_sw == 1], 1 - SENSITIVITY)
        ax.axvline(t90, color="k", ls=":", lw=1)
        ax.text(t90, 0.03, " порог «90% сна»", fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("порог на P(wake)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("значение")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "f1_threshold.png", dpi=130)
    plt.close(fig)

    # ---------------- рисунок 3: F1 по стадиям (три класса) + разброс F1(wake) по людям
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    classes = [("f1_wake", "wake", "#d62728"), ("f1_nrem", "NREM", "#4292c6"), ("f1_rem", "REM", "#2ca02c"),
               ("f1_macro", "macro F1", "#555555")]
    x = np.arange(len(tc))
    for i, (col, label, color) in enumerate(classes):
        axes[0].bar(x + (i - 1.5) * width, tc[col], width, label=label, color=color)
    axes[0].set_xticks(x, tc.model, fontsize=8, rotation=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Три класса: F1 по каждой стадии (argmax, LOSO)")
    axes[0].legend(ncol=4, fontsize=8)
    models = [m for m, _, _ in experiments]
    axes[1].boxplot([ps.loc[ps.model == m, "f1_wake"] for m in models], tick_labels=[m.replace(", ", "\n") for m in models])
    axes[1].set_title("F1 бодрствования по людям (31 человек)")
    axes[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "f1_three_class_and_subjects.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
