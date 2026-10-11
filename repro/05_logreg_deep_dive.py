"""Логистическая регрессия из статьи — подробный разбор (материал к защите).

    PYTHONPATH=. .venv/Scripts/python repro/05_logreg_deep_dive.py

Что делает:
  1. Обучает LR ровно как в коде статьи: liblinear, L1/L2 и C подбираются GridSearchCV(cv=3, roc_auc),
     class_weight='balanced', признаки БЕЗ стандартизации. Печатает уравнение модели и коэффициенты:
     сырые, на 1 SD признака, отношения шансов.
  2. 95% доверительные интервалы коэффициентов кластерным бутстрепом по людям.
  3. LOSO-оценка: AUC, Таблица-2-подобные метрики, AUC по людям, калибровка (и почему порог 0.5 ≠ «50%»).
  4. Путь регуляризации L1: как коэффициенты зануляются при уменьшении C.
  5. Проверка допущения линейности по log-odds (log1p(motion), сплайны) — простое улучшение LR.
  6. Три класса: one-vs-rest (как в статье, sklearn 0.20) против multinomial (softmax).
Результаты: outputs/repro/logreg/
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleeplib as sl  # noqa: E402
from source.utils import LiblinearLogisticRegression  # noqa: E402

warnings.filterwarnings("ignore")
N_JOBS = int(__import__("os").environ.get("SLEEP_N_JOBS", 8))
OUT = sl.OUT / "logreg"
OUT.mkdir(parents=True, exist_ok=True)
F = ["motion", "hr", "clock"]
GRID = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}  # ParameterSearch из кода статьи


def paper_lr():
    return LiblinearLogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced")


def fit_paper_lr(X, y, scoring="roc_auc"):
    """Как ClassifierService.train_classifier: веса классов 'balanced' + GridSearchCV(cv=3)."""
    g = GridSearchCV(paper_lr(), GRID, scoring=scoring, cv=3).fit(X, y)
    return g.best_estimator_, g.best_params_


def loso(data, make, transform=lambda X: X, target="sw"):
    conv = sl.sleep_wake if target == "sw" else sl.three_class

    def fold(test):
        X, y, _ = sl.stack(data, [s for s in sl.SUBJECTS if s != test], F)
        m = make().fit(transform(X), conv(y))
        return test, m.predict_proba(transform(sl.stack(data, [test], F)[0]))

    res = dict(Parallel(n_jobs=N_JOBS)(delayed(fold)(s) for s in sl.SUBJECTS))
    return np.vstack([res[s] for s in sl.SUBJECTS])


def main():
    data = sl.load_paper_features()
    X, y_raw, groups = sl.stack(data, sl.SUBJECTS, F)
    y = sl.sleep_wake(y_raw)
    report = {}

    # ------------------------------------------------------------ 1. модель статьи на всех данных
    model, params = fit_paper_lr(X, y)
    b0, b = float(model.intercept_[0]), model.coef_[0]
    sd = X.std(0)
    report["best_params"] = params
    report["equation"] = ("logit P(sleep) = %.3f %+.3f*motion %+.3f*hr %+.3f*clock" % (b0, *b))
    report["coef_raw"] = dict(zip(F, b.round(4)))
    report["coef_per_SD"] = dict(zip(F, (b * sd).round(4)))
    report["odds_ratio_per_SD"] = dict(zip(F, np.exp(b * sd).round(3)))
    report["feature_mean"] = dict(zip(F, X.mean(0).round(4)))
    report["feature_sd"] = dict(zip(F, sd.round(4)))
    print(report["equation"], params)

    # ------------------------------------------------------------ 2. кластерный бутстреп по людям
    rng = np.random.default_rng(0)
    subj_idx = {s: np.flatnonzero(groups == s) for s in sl.SUBJECTS}

    def boot(seed):
        r = np.random.default_rng(seed)
        pick = r.choice(sl.SUBJECTS, len(sl.SUBJECTS), replace=True)
        idx = np.concatenate([subj_idx[s] for s in pick])
        m = LiblinearLogisticRegression(solver="liblinear", class_weight="balanced", **params).fit(X[idx], y[idx])
        return np.r_[m.intercept_, m.coef_[0]]

    B = np.array(Parallel(n_jobs=N_JOBS)(delayed(boot)(int(s)) for s in rng.integers(0, 10**9, 300)))
    ci = np.percentile(B, [2.5, 97.5], axis=0)
    report["bootstrap_95ci"] = {n: [round(ci[0, i], 4), round(ci[1, i], 4)]
                                for i, n in enumerate(["intercept"] + F)}
    # «наивные» SE, как будто все 25 тыс. эпох независимы — для сравнения (они сильно занижены)
    p = model.predict_proba(X)[:, 1]
    w = model.class_weight
    sw = np.array([w[c] if isinstance(w, dict) else 1.0 for c in y]) if isinstance(w, dict) else np.where(
        y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
    Xd = np.c_[np.ones(len(X)), X]
    H = Xd.T @ (Xd * (sw * p * (1 - p))[:, None])
    naive_se = np.sqrt(np.diag(np.linalg.inv(H)))
    report["naive_se_iid"] = dict(zip(["intercept"] + F, naive_se.round(4)))
    report["bootstrap_se"] = dict(zip(["intercept"] + F, B.std(0).round(4)))

    # ------------------------------------------------------------ 3. LOSO качество + калибровка
    p_loso = loso(data, lambda: LiblinearLogisticRegression(solver="liblinear", class_weight="balanced",
                                                            **params))[:, 1]
    p_nowt = loso(data, lambda: LiblinearLogisticRegression(solver="liblinear", **params))[:, 1]
    report["loso"] = sl.sw_summary(y, p_loso)
    per_subj, i = [], 0
    threshold = np.quantile(p_loso[y == 1], 0.10)  # общий порог: сон найден = 0.90 по всем эпохам
    for s in sl.SUBJECTS:
        n = len(subj_idx[s])
        ys = y[i:i + n]
        pred = (p_loso[i:i + n] >= threshold).astype(int)
        has_wake = 0 < ys.mean() < 1
        f1 = sl.sw_f1(ys, pred)
        per_subj.append(dict(subject=s, n=n, wake_frac=1 - ys.mean(),
                             auc=roc_auc_score(ys, p_loso[i:i + n]) if has_wake else np.nan,
                             accuracy=np.mean(pred == ys),
                             wake_correct=np.mean(pred[ys == 0] == 0) if has_wake else np.nan,
                             sleep_correct=np.mean(pred[ys == 1] == 1),
                             kappa=cohen_kappa_score(ys, pred) if has_wake else np.nan,
                             f1_wake=f1["f1_wake"] if has_wake else np.nan,
                             f1_macro=f1["f1_macro"] if has_wake else np.nan))
        i += n
    per_subj = pd.DataFrame(per_subj)
    per_subj.to_csv(OUT / "loso_per_subject.csv", index=False, float_format="%.4f")
    yy = np.concatenate([y[subj_idx[s]] for s in sl.SUBJECTS])  # тот же порядок, что у p_loso
    assert (yy == y).all()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    # (a) линейный счёт и сигмоида
    z = np.r_[np.ones((len(X), 1)).T, X.T].T @ np.r_[b0, b]
    ax = axes[0, 0]
    bins = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 60)
    ax.hist(z[y == 1], bins=bins, alpha=.6, density=True, label="sleep (PSG)", color="#4292c6")
    ax.hist(z[y == 0], bins=bins, alpha=.6, density=True, label="wake (PSG)", color="#d62728")
    ax2 = ax.twinx()
    zz = np.linspace(bins[0], bins[-1], 200)
    ax2.plot(zz, 1 / (1 + np.exp(-zz)), "k")
    ax2.set_ylabel("σ(z) = P(sleep)")
    ax.set_xlabel("z = β0 + β·x  (log-odds сна)")
    ax.set_title("Линейный счёт по классам и сигмоида")
    ax.legend(loc="upper left")
    # (b) вклад motion при разных hr / clock
    ax = axes[0, 1]
    mgrid = np.linspace(0, np.percentile(X[:, 0], 99), 200)
    for q, ls in [(0.1, ":"), (0.5, "-"), (0.9, "--")]:
        for f, col in [("clock", "#d3494e"), ("hr", "#fcb001")]:
            xx = np.tile(np.median(X, 0), (200, 1))
            xx[:, 0] = mgrid
            xx[:, F.index(f)] = np.quantile(X[:, F.index(f)], q)
            ax.plot(mgrid, model.predict_proba(xx)[:, 1], ls=ls, color=col,
                    label=f"{f} = q{int(q * 100)}" if True else None)
    ax.set_xlabel("motion (activity counts, сглаж.)")
    ax.set_ylabel("P(sleep) модели")
    ax.set_title("P(sleep) vs motion при разных hr / clock\n(прочие признаки — медиана)")
    ax.legend(fontsize=8)
    # (c) калибровка
    ax = axes[0, 2]
    for pp, name in [(p_loso, "class_weight='balanced' (как в статье)"), (p_nowt, "без весов классов")]:
        fr, mp = calibration_curve(y, pp, n_bins=15, strategy="quantile")
        ax.plot(mp, fr, "o-", ms=3, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("предсказанная P(sleep)")
    ax.set_ylabel("реальная доля sleep")
    ax.set_title("Калибровка (LOSO)")
    ax.legend(fontsize=8)
    # (d) AUC по людям
    ax = axes[1, 0]
    ax.scatter(per_subj.wake_frac, per_subj.auc, c="k", s=18)
    for r in per_subj.itertuples():
        ax.annotate(r.subject, (r.wake_frac, r.auc), fontsize=6, alpha=.6)
    ax.set_xlabel("доля wake у человека")
    ax.set_ylabel("AUC sleep/wake (модель без этого человека)")
    ax.set_title(f"Разброс по людям: медиана AUC = {per_subj.auc.median():.3f}")
    # (e) путь регуляризации L1
    ax = axes[1, 1]
    Cs = np.logspace(-4, 2, 40)
    path = np.array([LogisticRegression(penalty="l1", solver="liblinear", C=c, class_weight="balanced")
                    .fit(X, y).coef_[0] * sd for c in Cs])
    for j, f in enumerate(F):
        ax.plot(Cs, path[:, j], label=f)
    ax.axvline(params["C"], color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("C (меньше — сильнее L1-штраф)")
    ax.set_ylabel("коэффициент × SD признака")
    ax.set_title("Путь регуляризации L1")
    ax.legend()
    # (f) бутстреп-распределения коэффициентов
    ax = axes[1, 2]
    ax.boxplot([B[:, j + 1] * sd[j] for j in range(3)], tick_labels=F)
    ax.axhline(0, color="k", lw=.5)
    ax.set_ylabel("коэффициент × SD признака")
    ax.set_title("Бутстреп по людям (300 повторов)")
    fig.tight_layout()
    fig.savefig(OUT / "logreg_overview.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------ 5. линейность по log-odds
    def lr_scaled():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))

    log_motion = lambda A: np.c_[np.log1p(A[:, 0]), A[:, 1:]]

    def lr_spline():
        return make_pipeline(SplineTransformer(n_knots=6, degree=3), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))

    variants = {
        "LR статьи (сырые признаки, L1/L2+GridSearch)": p_loso,
        "LR + StandardScaler": loso(data, lr_scaled)[:, 1],
        "LR + log1p(motion)": loso(data, lr_scaled, log_motion)[:, 1],
        "LR + сплайны по каждому признаку (GAM-подобная)": loso(data, lr_spline, log_motion)[:, 1],
    }
    lin = pd.DataFrame({k: sl.sw_summary(y, v) for k, v in variants.items()}).T
    lin.to_csv(OUT / "linearity_variants.csv", float_format="%.4f")
    report["linearity_variants"] = lin[["auc", "accuracy@90", "wake_correct@90", "kappa@90", "f1_wake@90",
                                        "f1_macro@90", "accuracy@93", "wake_correct@93", "kappa@93"]].round(4).to_dict(
        orient="index")

    # форма зависимости log-odds от каждого признака (по биннингу) — видно, где линейность нарушена
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, f in enumerate(F):
        xv = log_motion(X)[:, 0] if f == "motion" else X[:, j]
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, 21)))
        idx = np.clip(np.digitize(xv, qs[1:-1]), 0, len(qs) - 2)
        centers = [xv[idx == k].mean() for k in range(len(qs) - 1)]
        rate = np.array([y[idx == k].mean() for k in range(len(qs) - 1)])
        rate = np.clip(rate, 1e-3, 1 - 1e-3)
        axes[j].plot(centers, np.log(rate / (1 - rate)), "o-")
        axes[j].set_xlabel("log1p(motion)" if f == "motion" else f)
        axes[j].set_ylabel("эмпирический logit P(sleep)")
        axes[j].set_title(f"{f}: линейна ли зависимость?")
    fig.tight_layout()
    fig.savefig(OUT / "logreg_linearity_check.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------ 6. три класса: OvR (статья) vs multinomial
    y3 = sl.three_class(y_raw)
    m3, params3 = fit_paper_lr(X, y3, scoring="neg_log_loss")
    report["three_class_best_params"] = params3
    report["three_class_ovr_coef_raw"] = {c: dict(zip(F, m3.coef_[k].round(4)))
                                          for k, c in enumerate(["wake", "nrem", "rem"])}
    report["three_class_ovr_intercepts"] = dict(zip(["wake", "nrem", "rem"], m3.intercept_.round(4)))
    p3_ovr = loso(data, lambda: LiblinearLogisticRegression(solver="liblinear", class_weight="balanced",
                                                            **params3), target="3")
    p3_mn = loso(data, lambda: make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000)), target="3")
    report["three_class"] = {
        "ovr_liblinear (как в статье)": sl.three_class_at_wake_fpr(y3, p3_ovr, 0.4),
        "multinomial (softmax)": sl.three_class_at_wake_fpr(y3, p3_mn, 0.4),
    }
    (OUT / "logreg_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                                            encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
