"""Попытка улучшить модель статьи — на тех же данных, тех же эпохах и тех же метриках (LOSO, 31 фолд).

Сравниваем:
  1. paper-MLP / paper-LR   — модели статьи на (motion, hr, clock)
  2. LGBM paper             — градиентный бустинг на тех же 3 признаках
  3. LR / MLP / LGBM ext    — на расширенных признаках (ext_features.py)
  4. LGBM ext + HMM         — + сглаживание последовательности стадий (forward–backward HMM)
  5. Абляция групп расширенных признаков (LightGBM)
Плюс SHAP для лучшей модели — какие из новых признаков реально работают.

Результаты: outputs/repro/improve/
"""
import os
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ext_features as ef
import sleeplib as sl

warnings.filterwarnings("ignore")
OUT = sl.OUT / "improve"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0


def model(kind, n_classes):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(15, 15, 15), alpha=0.01,
                                                             max_iter=2000, n_iter_no_change=20, random_state=SEED))
    if kind == "mlp_big":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                             early_stopping=True, max_iter=500, random_state=SEED))
    if kind == "lgbm":
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                  random_state=SEED, verbose=-1, n_jobs=1)
    raise ValueError(kind)


# ------------------------------------------------------------------ HMM smoothing (forward–backward)
def transition_matrix(label_seqs, k, pseudo=1.0):
    T = np.full((k, k), pseudo)
    start = np.full(k, pseudo)
    for y in label_seqs:
        start[y[0]] += 1
        np.add.at(T, (y[:-1], y[1:]), 1)
    return T / T.sum(1, keepdims=True), start / start.sum()


def hmm_smooth(proba, prior, T, start):
    e = proba / prior
    n, k = e.shape
    alpha = np.zeros((n, k))
    beta = np.ones((n, k))
    a = start * e[0]
    alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * e[t]
        alpha[t] = a / a.sum()
    for t in range(n - 2, -1, -1):
        b = T @ (e[t + 1] * beta[t + 1])
        beta[t] = b / b.sum()
    post = alpha * beta
    return post / post.sum(1, keepdims=True)


# ------------------------------------------------------------------ LOSO evaluation
def run(Xd, yd, cols, kind, target, smooth=False):
    k = 2 if target == "sw" else 3
    conv = sl.sleep_wake if target == "sw" else sl.three_class

    def fold(test):
        train = [s for s in sl.SUBJECTS if s != test]
        X = np.vstack([Xd[s][:, cols] for s in train])
        y = np.concatenate([conv(yd[s]) for s in train])
        m = model(kind, k).fit(X, y)
        p = m.predict_proba(Xd[test][:, cols])
        if smooth:
            T, start = transition_matrix([conv(yd[s]) for s in train], k)
            p = hmm_smooth(p, np.bincount(y, minlength=k) / len(y), T, start)
        return test, p

    res = dict(Parallel(n_jobs=int(os.environ.get("SLEEP_N_JOBS", -1)))(delayed(fold)(s) for s in sl.SUBJECTS))
    return {s: res[s] for s in sl.SUBJECTS}


def evaluate(preds, yd):
    p = np.vstack([preds[s] for s in sl.SUBJECTS])
    y = np.concatenate([yd[s] for s in sl.SUBJECTS])
    if p.shape[1] == 2:
        m = sl.sw_summary(sl.sleep_wake(y), p[:, 1])
        per = [roc_auc_score(sl.sleep_wake(yd[s]), preds[s][:, 1]) for s in sl.SUBJECTS
               if 0 < sl.sleep_wake(yd[s]).mean() < 1]
        m["auc_subject_median"] = np.median(per)
        return m
    y3 = sl.three_class(y)
    r = sl.three_class_at_wake_fpr(y3, p, 0.4)
    pred = p.argmax(1)
    return dict(nrem_correct=r["nrem_correct"], rem_correct=r["rem_correct"], best_accuracy=r["best_accuracy"],
                best_kappa=r["best_kappa"], balanced_accuracy=r["balanced_accuracy"],
                balanced_kappa=r["balanced_kappa"], balanced_f1_macro=r["balanced_f1_macro"],
                argmax_accuracy=np.mean(pred == y3), argmax_kappa=cohen_kappa_score(y3, pred),
                argmax_f1_macro=sl.three_class_f1(y3, pred)["f1_macro"],
                rem_auc=roc_auc_score(y3 == 2, p[:, 2]), wake_auc=roc_auc_score(y3 == 0, p[:, 0]))


def main():
    paper = sl.load_paper_features()
    print("building extended features...", flush=True)
    Xd, names = ef.load_all(paper)
    yd = {s: paper[s]["stage"] for s in sl.SUBJECTS}
    idx = {n: i for i, n in enumerate(names)}
    paper_cols = [idx["paper_motion"], idx["paper_hr"], idx["paper_clock"]]
    all_cols = list(range(len(names)))

    experiments = [
        ("paper-LR (motion,hr,clock)", paper_cols, "lr", False),
        ("paper-MLP (motion,hr,clock)", paper_cols, "mlp", False),
        ("LGBM (motion,hr,clock)", paper_cols, "lgbm", False),
        ("LR ext", all_cols, "lr", False),
        ("MLP ext", all_cols, "mlp_big", False),
        ("LGBM ext", all_cols, "lgbm", False),
        ("LGBM (motion,hr,clock) + HMM", paper_cols, "lgbm", True),
        ("LGBM ext + HMM", all_cols, "lgbm", True),
    ]
    rows_sw, rows_3, preds_keep = [], [], {}
    for name, cols, kind, smooth in experiments:
        for target, rows in [("sw", rows_sw), ("3", rows_3)]:
            preds = run(Xd, yd, cols, kind, target, smooth)
            m = evaluate(preds, yd)
            rows.append(dict(model=name, **m))
            preds_keep[(name, target)] = preds
            key = "auc" if target == "sw" else "best_kappa"
            print(f"  {target:2s} {name:32s} {key}={m[key]:.3f}", flush=True)
    sw = pd.DataFrame(rows_sw)
    th = pd.DataFrame(rows_3)
    sw.to_csv(OUT / "sleep_wake_comparison.csv", index=False, float_format="%.4f")
    th.to_csv(OUT / "three_class_comparison.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(sw[["model", "auc", "auc_subject_median", "wake_correct@90", "accuracy@90", "kappa@90",
              "wake_correct@93", "kappa@93"]].round(3).to_string(index=False))
    print(th.round(3).to_string(index=False))

    # ---------------- абляция групп признаков (LightGBM, sleep/wake + 3 класса)
    groups = {
        "paper": [n for n in names if n.startswith("paper_")],
        "counts": [n for n in names if n.startswith("counts_")],
        "accel": [n for n in names if n.startswith(("enmo_", "dangle_", "active_frac_", "log_"))],
        "hrlvl": [n for n in names if n.startswith(("hr_level", "hr_std", "hr_trend"))],
        "time": ["time_h"],
    }
    abl = []
    combos = [["paper"], ["paper", "counts"], ["paper", "accel"], ["paper", "hrlvl"], ["paper", "time"],
              ["paper", "counts", "accel", "hrlvl"], ["paper", "counts", "accel", "hrlvl", "time"],
              ["counts", "accel", "hrlvl"], ["counts", "accel", "hrlvl", "time"]]
    for combo in combos:
        cols = [idx[n] for g in combo for n in groups[g]]
        msw = evaluate(run(Xd, yd, cols, "lgbm", "sw"), yd)
        m3 = evaluate(run(Xd, yd, cols, "lgbm", "3"), yd)
        abl.append(dict(groups="+".join(combo), n_features=len(cols), auc=msw["auc"],
                        accuracy_90=msw["accuracy@90"], wake_correct_90=msw["wake_correct@90"],
                        kappa_90=msw["kappa@90"], f1_wake_90=msw["f1_wake@90"], f1_macro_90=msw["f1_macro@90"],
                        three_best_kappa=m3["best_kappa"], three_best_acc=m3["best_accuracy"],
                        three_argmax_acc=m3["argmax_accuracy"], three_argmax_f1_macro=m3["argmax_f1_macro"],
                        rem_auc=m3["rem_auc"]))
        print("  ablation", abl[-1], flush=True)
    abl = pd.DataFrame(abl)
    abl.to_csv(OUT / "ablation_groups.csv", index=False, float_format="%.4f")
    print(abl.round(3).to_string(index=False))

    # ---------------- ROC: модель статьи vs улучшенные
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    grid = np.linspace(0, 1, 201)
    y_all = np.concatenate([yd[s] for s in sl.SUBJECTS])
    for name, color in [("paper-MLP (motion,hr,clock)", "#d3494e"), ("LGBM ext", "#2a7ab9"),
                        ("LGBM ext + HMM", "#111111")]:
        p = np.vstack([preds_keep[(name, "sw")][s] for s in sl.SUBJECTS])[:, 1]
        axes[0].plot(grid, sl.interp_roc(sl.sleep_wake(y_all), p, grid), color=color, label=name)
        p3 = np.vstack([preds_keep[(name, "3")][s] for s in sl.SUBJECTS])
        curve = [sl.three_class_at_wake_fpr(sl.three_class(y_all), p3, f) for f in np.linspace(0.05, 0.95, 19)]
        axes[1].plot([c["wake_fpr"] for c in curve], [(c["nrem_correct"] + c["rem_correct"]) / 2 for c in curve],
                     color=color, label=name)
    axes[0].set_xlabel("доля wake, названная sleep")
    axes[0].set_ylabel("доля sleep, названная sleep")
    axes[0].set_title("Sleep/wake ROC (LOSO, все эпохи)")
    axes[1].set_xlabel("доля wake, названная REM/NREM")
    axes[1].set_ylabel("средняя доля верных NREM и REM (уравненных)")
    axes[1].set_title("Wake/NREM/REM (как рис. 4 статьи)")
    for ax in axes:
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "roc_paper_vs_improved.png", dpi=130)
    plt.close(fig)

    # ---------------- SHAP лучшей модели (LGBM ext) — обучаем на всех, смотрим на выборке
    X = np.vstack([Xd[s] for s in sl.SUBJECTS])
    m = model("lgbm", 2).fit(X, sl.sleep_wake(y_all))
    sample = np.random.default_rng(SEED).choice(len(X), 5000, replace=False)
    sv = shap.TreeExplainer(m).shap_values(X[sample])
    sv = sv[..., 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    plt.figure()
    shap.summary_plot(sv, X[sample], feature_names=names, max_display=20, show=False, plot_size=(9, 8))
    plt.title("SHAP, LGBM ext (sleep/wake), log-odds сна")
    plt.tight_layout()
    plt.savefig(OUT / "shap_lgbm_ext_sw.png", dpi=130)
    plt.close("all")
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    imp.to_csv(OUT / "shap_lgbm_ext_sw_mean_abs.csv", float_format="%.4f")
    print(imp.head(15).round(3))

    m3 = model("lgbm", 3).fit(X, sl.three_class(y_all))
    sv3 = shap.TreeExplainer(m3).shap_values(X[sample])
    sv3 = np.stack(sv3, -1) if isinstance(sv3, list) else sv3
    imp3 = pd.DataFrame(np.abs(sv3).mean(0), index=names, columns=["wake", "NREM", "REM"])
    imp3.sort_values("REM", ascending=False).to_csv(OUT / "shap_lgbm_ext_3class_mean_abs.csv", float_format="%.4f")
    plt.figure()
    shap.summary_plot(sv3[..., 2], X[sample], feature_names=names, max_display=15, show=False, plot_size=(9, 7))
    plt.title("SHAP для класса REM (LGBM ext, 3 класса)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_lgbm_ext_rem.png", dpi=130)
    plt.close("all")
    print(imp3.sort_values("REM", ascending=False).head(10).round(3))


if __name__ == "__main__":
    main()
