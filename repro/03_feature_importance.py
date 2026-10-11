"""Важность признаков модели статьи (motion, hr, clock) — четырьмя независимыми способами.

  A. «Добавочная» важность: качество (AUC, κ) для ВСЕХ 7 подмножеств признаков + точное разложение
     Шепли прироста AUC (сколько AUC «приносит» каждый признак в среднем по всем порядкам добавления),
     а также drop-column (сколько теряем, если убрать признак из полной модели).
  B. Веса моделей: стандартизованные коэффициенты логистической регрессии (знак = направление),
     MDI (impurity) важность случайного леса.
  C. Permutation importance на отложенных людях (LOSO): падение AUC при перемешивании признака.
  D. SHAP: TreeSHAP для случайного леса и LightGBM, PermutationSHAP для нейросети (MLP);
     beeswarm, зависимости, вклад признаков вдоль ночи, SHAP для трёх классов.

Все оценки — leave-one-subject-out (31 фолд): модель никогда не видит человека, на котором её проверяют.
Результаты: outputs/repro/importance/
"""
import itertools
import json
import math
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sleeplib as sl

warnings.filterwarnings("ignore")
OUT = sl.OUT / "importance"
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ["motion", "hr", "clock"]
NAMES_RU = {"motion": "motion (activity counts)", "hr": "hr (лок. вариабельность ЧСС)", "clock": "clock proxy"}
SEED = 0


def make_model(kind, n_classes=2):
    cw = "balanced"
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight=cw, max_iter=2000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=32,
                                      max_features=1.0, class_weight=cw, n_jobs=1, random_state=SEED)
    if kind == "mlp":  # как в utils.get_classifiers(); MLP в sklearn не поддерживает class_weight
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(15, 15, 15), alpha=0.01, max_iter=2000,
                                           n_iter_no_change=20, random_state=SEED))
    if kind == "lgbm":
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=1.0, class_weight=cw,
                                  random_state=SEED, verbose=-1, n_jobs=1)
    raise ValueError(kind)


def loso_predict(data, features, kind, target="sw"):
    """Out-of-fold вероятности для всех эпох (LOSO)."""
    def fold(test_subject):
        train = [s for s in sl.SUBJECTS if s != test_subject]
        X, y, _ = sl.stack(data, train, features)
        Xt, yt, _ = sl.stack(data, [test_subject], features)
        y = sl.sleep_wake(y) if target == "sw" else sl.three_class(y)
        model = make_model(kind).fit(X, y)
        return test_subject, model.predict_proba(Xt)

    results = dict(Parallel(n_jobs=-1)(delayed(fold)(s) for s in sl.SUBJECTS))
    return np.vstack([results[s] for s in sl.SUBJECTS])


def labels(data, target="sw"):
    y = np.concatenate([data[s]["stage"] for s in sl.SUBJECTS])
    return sl.sleep_wake(y) if target == "sw" else sl.three_class(y)


def per_subject_auc(data, p_sleep):
    out, i = [], 0
    for s in sl.SUBJECTS:
        n = len(data[s]["stage"])
        y = sl.sleep_wake(data[s]["stage"])
        out.append(roc_auc_score(y, p_sleep[i:i + n]) if 0 < y.mean() < 1 else np.nan)
        i += n
    return np.array(out)


# --------------------------------------------------------------------------- A. additive / Shapley over subsets
def subset_analysis(data):
    y = labels(data)
    rows = []
    for kind in ["lr", "rf", "mlp"]:
        for r in range(1, 4):
            for subset in itertools.combinations(FEATURES, r):
                p = loso_predict(data, list(subset), kind)[:, 1]
                m = sl.sw_summary(y, p)
                rows.append(dict(model=kind, subset="+".join(subset), k=r, **m,
                                 auc_subject_mean=np.nanmean(per_subject_auc(data, p))))
                print(f"  {kind:4s} {'+'.join(subset):18s} AUC={m['auc']:.3f} wake@90={m['wake_correct@90']:.3f}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A_subsets.csv", index=False, float_format="%.4f")

    # Точное значение Шепли для «игры» v(S) = AUC(S) − 0.5 (v(∅)=0, AUC случайной модели = 0.5)
    shapley_rows = []
    for kind in ["lr", "rf", "mlp"]:
        v = {frozenset(): 0.0}
        for _, r in df[df.model == kind].iterrows():
            v[frozenset(r.subset.split("+"))] = r.auc - 0.5
        n = len(FEATURES)
        for f in FEATURES:
            others = [g for g in FEATURES if g != f]
            phi = 0.0
            for r in range(0, n):
                for S in itertools.combinations(others, r):
                    w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                    phi += w * (v[frozenset(S) | {f}] - v[frozenset(S)])
            full = v[frozenset(FEATURES)]
            drop = full - v[frozenset(others)]
            alone = v[frozenset([f])]
            shapley_rows.append(dict(model=kind, feature=f, shapley_auc=phi, drop_column_auc=drop,
                                     alone_auc=alone, share=phi / full))
    sh = pd.DataFrame(shapley_rows)
    sh.to_csv(OUT / "A_shapley_auc.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    pivot = df.pivot(index="subset", columns="model", values="auc").loc[
        ["motion", "hr", "clock", "motion+hr", "motion+clock", "hr+clock", "motion+hr+clock"]]
    pivot[["lr", "rf", "mlp"]].plot.barh(ax=axes[0], color=["#4c72b0", "#55a868", "#c44e52"])
    axes[0].set_xlim(0.5, 0.92)
    axes[0].set_xlabel("AUC sleep/wake (LOSO)")
    axes[0].set_title("Качество на всех подмножествах признаков")
    piv = sh.pivot(index="feature", columns="model", values="shapley_auc").loc[FEATURES][["lr", "rf", "mlp"]]
    piv.plot.bar(ax=axes[1], color=["#4c72b0", "#55a868", "#c44e52"], rot=0)
    axes[1].set_ylabel("вклад в (AUC − 0.5)")
    axes[1].set_title("Аддитивная важность: значение Шепли по AUC")
    fig.tight_layout()
    fig.savefig(OUT / "A_subsets_shapley.png", dpi=130)
    plt.close(fig)
    return df, sh


# --------------------------------------------------------------------------- B. model weights
def model_weights(data):
    X, y, _ = sl.stack(data, sl.SUBJECTS, FEATURES)
    ysw, y3 = sl.sleep_wake(y), sl.three_class(y)
    rows = []
    # Распределение коэффициентов по LOSO-фолдам — показывает устойчивость весов
    for test in sl.SUBJECTS:
        train = [s for s in sl.SUBJECTS if s != test]
        Xtr, ytr, _ = sl.stack(data, train, FEATURES)
        lr = make_model("lr").fit(Xtr, sl.sleep_wake(ytr))
        coefs = lr[-1].coef_[0]
        rows.append(dict(zip(FEATURES, coefs)))
    coefs = pd.DataFrame(rows)
    lr_full = make_model("lr").fit(X, ysw)
    lr3 = make_model("lr").fit(X, y3)
    rf = make_model("rf").fit(X, ysw)
    weights = dict(
        lr_sleep_vs_wake=dict(zip(FEATURES, lr_full[-1].coef_[0].round(4))),
        lr_intercept=float(lr_full[-1].intercept_[0]),
        lr_three_class={c: dict(zip(FEATURES, lr3[-1].coef_[i].round(4))) for i, c in enumerate(["wake", "nrem", "rem"])},
        rf_mdi=dict(zip(FEATURES, rf.feature_importances_.round(4))),
        lr_coef_loso_mean=coefs.mean().round(4).to_dict(), lr_coef_loso_std=coefs.std().round(4).to_dict(),
        feature_means=dict(zip(FEATURES, X.mean(0).round(4))), feature_stds=dict(zip(FEATURES, X.std(0).round(4))),
    )
    (OUT / "B_weights.json").write_text(json.dumps(weights, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].boxplot([coefs[f] for f in FEATURES], tick_labels=FEATURES)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("LR: станд. коэффициенты (sleep=1)\nпо 31 LOSO-фолду")
    axes[0].set_ylabel("Δ log-odds сна на 1 SD признака")
    c3 = pd.DataFrame(weights["lr_three_class"]).T[FEATURES]
    c3.plot.bar(ax=axes[1], rot=0, color=["#3b5b92", "#fcb001", "#d3494e"])
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("LR 3 класса: коэффициенты по классам")
    axes[2].bar(FEATURES, rf.feature_importances_, color="#55a868")
    axes[2].set_title("Random forest: MDI (impurity) важность")
    fig.tight_layout()
    fig.savefig(OUT / "B_model_weights.png", dpi=130)
    plt.close(fig)
    return weights


# --------------------------------------------------------------------------- C. permutation importance (LOSO)
def permutation_importance(data, kind, n_repeats=10):
    rng = np.random.default_rng(SEED)

    def fold(test_subject):
        train = [s for s in sl.SUBJECTS if s != test_subject]
        X, y, _ = sl.stack(data, train, FEATURES)
        Xt, yt, _ = sl.stack(data, [test_subject], FEATURES)
        return test_subject, make_model(kind).fit(X, sl.sleep_wake(y))

    models = dict(Parallel(n_jobs=-1)(delayed(fold)(s) for s in sl.SUBJECTS))
    Xs = {s: sl.stack(data, [s], FEATURES)[0] for s in sl.SUBJECTS}
    y = labels(data)

    def predict(Xs_):
        return np.concatenate([models[s].predict_proba(Xs_[s])[:, 1] for s in sl.SUBJECTS])

    base = roc_auc_score(y, predict(Xs))
    rows = []
    for j, f in enumerate(FEATURES):
        for mode in ["global", "within_subject"]:
            drops = []
            for _ in range(n_repeats):
                Xp = {s: Xs[s].copy() for s in sl.SUBJECTS}
                if mode == "within_subject":  # перемешиваем внутри ночи: разрушаем ТОЛЬКО временную информацию
                    for s in sl.SUBJECTS:
                        Xp[s][:, j] = rng.permutation(Xp[s][:, j])
                else:  # перемешиваем по всем людям: разрушаем и межличностную, и временную информацию
                    allv = rng.permutation(np.concatenate([Xs[s][:, j] for s in sl.SUBJECTS]))
                    i = 0
                    for s in sl.SUBJECTS:
                        n = len(Xp[s])
                        Xp[s][:, j] = allv[i:i + n]
                        i += n
                drops.append(base - roc_auc_score(y, predict(Xp)))
            rows.append(dict(model=kind, feature=f, mode=mode, auc_drop_mean=np.mean(drops),
                             auc_drop_std=np.std(drops), base_auc=base))
    return rows, models


# --------------------------------------------------------------------------- D. SHAP
def shap_analysis(data, models_by_kind):
    """SHAP на out-of-fold данных: для каждого человека используем модель, обученную БЕЗ него."""
    rng = np.random.default_rng(SEED)
    sample_idx = {s: np.sort(rng.choice(len(data[s]["stage"]), size=min(150, len(data[s]["stage"])), replace=False))
                  for s in sl.SUBJECTS}
    result = {}
    for kind in ["rf", "lgbm", "mlp"]:
        models = models_by_kind[kind]
        values, Xall = [], []
        for s in sl.SUBJECTS:
            Xs = sl.stack(data, [s], FEATURES)[0][sample_idx[s]]
            model = models[s]
            if kind in ("rf", "lgbm"):
                sv = shap.TreeExplainer(model).shap_values(Xs)
                sv = sv[..., 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
            else:
                train = [t for t in sl.SUBJECTS if t != s]
                bg = shap.sample(sl.stack(data, train, FEATURES)[0], 100, random_state=SEED)
                f = lambda x, m=model: m.predict_proba(x)[:, 1]
                sv = shap.PermutationExplainer(f, bg)(Xs).values
            values.append(sv)
            Xall.append(Xs)
        values, Xall = np.vstack(values), np.vstack(Xall)
        result[kind] = (values, Xall)
        np.savez(OUT / f"D_shap_{kind}.npz", values=values, X=Xall)

        unit = "log-odds" if kind == "lgbm" else ("P(sleep)" if kind == "mlp" else "P(sleep)")
        plt.figure()
        shap.summary_plot(values, Xall, feature_names=FEATURES, show=False, plot_size=(8, 3.6))
        plt.title(f"SHAP ({kind.upper()}, sleep/wake, out-of-fold), единицы: {unit}")
        plt.tight_layout()
        plt.savefig(OUT / f"D_shap_beeswarm_{kind}.png", dpi=130)
        plt.close("all")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for j, f in enumerate(FEATURES):
            other = FEATURES[(j + 1) % 3] if f != "motion" else "clock"
            k = FEATURES.index(other)
            sc = axes[j].scatter(Xall[:, j], values[:, j], c=Xall[:, k], s=4, cmap="coolwarm",
                                 vmin=np.quantile(Xall[:, k], 0.05), vmax=np.quantile(Xall[:, k], 0.95))
            axes[j].axhline(0, color="k", lw=0.5)
            axes[j].set_xlabel(NAMES_RU[f])
            axes[j].set_ylabel(f"SHAP({f})")
            if f == "motion":
                axes[j].set_xscale("symlog", linthresh=1)
            plt.colorbar(sc, ax=axes[j], label=other)
        fig.suptitle(f"SHAP dependence ({kind.upper()}): >0 — признак толкает к «сну», <0 — к «бодрствованию»")
        fig.tight_layout()
        fig.savefig(OUT / f"D_shap_dependence_{kind}.png", dpi=130)
        plt.close(fig)
    return result


def shap_over_night(data, models, subject):
    """Вклад признаков в P(sleep) вдоль одной ночи (LightGBM, модель без этого человека)."""
    Xs, ys, _ = sl.stack(data, [subject], FEATURES)
    model = models[subject]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    sv = sv[..., 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    t = np.arange(len(ys)) * 0.5 / 60
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    colors = {"motion": "#3b5b92", "hr": "#fcb001", "clock": "#d3494e"}
    for j, f in enumerate(FEATURES):
        axes[0].plot(t, sv[:, j], color=colors[f], lw=1, label=f)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("SHAP, log-odds сна")
    axes[0].legend(ncol=3)
    axes[0].set_title(f"Испытуемый {subject}: что «говорит» каждый признак по ходу ночи (LightGBM, out-of-fold)")
    p = model.predict_proba(Xs)[:, 1]
    axes[1].plot(t, 1 - p, color="k", lw=0.8)
    axes[1].set_ylabel("P(wake)")
    order = {0: 4, 5: 3, 1: 2, 2: 1, 3: 0, 4: 0}
    axes[2].step(t, [order[int(v)] for v in ys], where="post", color="k", lw=0.8)
    axes[2].set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "Wake"])
    axes[2].set_xlabel("часы от первой валидной эпохи")
    fig.tight_layout()
    fig.savefig(OUT / f"D_shap_over_night_{subject}.png", dpi=130)
    plt.close(fig)


def shap_three_class(data):
    X, y, _ = sl.stack(data, sl.SUBJECTS, FEATURES)
    model = make_model("lgbm").fit(X, sl.three_class(y))
    idx = np.random.default_rng(SEED).choice(len(X), 4000, replace=False)
    sv = shap.TreeExplainer(model).shap_values(X[idx])
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv  # (n, features, classes)
    mean_abs = np.abs(sv).mean(0)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    pd.DataFrame(mean_abs, index=FEATURES, columns=["wake", "NREM", "REM"]).plot.bar(
        ax=axes[0], rot=0, color=["#d62728", "#4292c6", "#2ca02c"])
    axes[0].set_title("mean |SHAP| по классам (LightGBM, 3 класса)")
    for k, (f, c, name) in enumerate([("hr", 2, "REM"), ("motion", 0, "wake"), ("clock", 2, "REM")]):
        j = FEATURES.index(f)
        axes[k + 1].scatter(X[idx, j], sv[:, j, c], s=3, c=X[idx, FEATURES.index("clock")], cmap="coolwarm")
        axes[k + 1].axhline(0, color="k", lw=0.5)
        axes[k + 1].set_xlabel(f)
        axes[k + 1].set_ylabel(f"SHAP({f}) для класса {name}")
        if f == "motion":
            axes[k + 1].set_xscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(OUT / "D_shap_three_class.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(mean_abs, index=FEATURES, columns=["wake", "NREM", "REM"])


def feature_overview(data):
    X, y, _ = sl.stack(data, sl.SUBJECTS, FEATURES + ["cosine", "time"])
    df = pd.DataFrame(X, columns=FEATURES + ["cosine", "time"])
    df["stage"] = pd.Series(sl.three_class(y)).map({0: "Wake", 1: "NREM", 2: "REM"})
    corr = df.drop(columns="stage").corr(method="spearman")
    corr.to_csv(OUT / "features_spearman_corr.csv", float_format="%.3f")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, f in zip(axes[:3], FEATURES):
        vals = [df.loc[df.stage == s, f].values for s in ["Wake", "NREM", "REM"]]
        if f == "motion":
            vals = [np.log10(1 + v) for v in vals]
        ax.violinplot(vals, showmedians=True)
        ax.set_xticks([1, 2, 3], ["Wake", "NREM", "REM"])
        ax.set_title(("log10(1+motion)" if f == "motion" else f) + " по стадиям")
    im = axes[3].imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    axes[3].set_xticks(range(5), corr.columns, rotation=45)
    axes[3].set_yticks(range(5), corr.columns)
    for i in range(5):
        for j in range(5):
            axes[3].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[3].set_title("Spearman-корреляции признаков")
    plt.colorbar(im, ax=axes[3])
    fig.tight_layout()
    fig.savefig(OUT / "features_overview.png", dpi=130)
    plt.close(fig)
    return corr


def main():
    data = sl.load_paper_features()
    print("features overview...", flush=True)
    corr = feature_overview(data)
    print(corr.round(2))
    print("A. subsets + Shapley...", flush=True)
    _, sh = subset_analysis(data)
    print(sh.round(3).to_string(index=False))
    print("B. model weights...", flush=True)
    print(json.dumps(model_weights(data), indent=1, ensure_ascii=False))
    print("C. permutation importance...", flush=True)
    rows, models_by_kind = [], {}
    for kind in ["lr", "rf", "mlp", "lgbm"]:
        r, models_by_kind[kind] = permutation_importance(data, kind)
        rows += r
    perm = pd.DataFrame(rows)
    perm.to_csv(OUT / "C_permutation.csv", index=False, float_format="%.4f")
    print(perm.round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, mode in zip(axes, ["global", "within_subject"]):
        pv = perm[perm["mode"] == mode].pivot(index="feature", columns="model", values="auc_drop_mean").loc[FEATURES]
        pv[["lr", "rf", "mlp", "lgbm"]].plot.bar(ax=ax, rot=0)
        ax.set_title("перемешивание по всем людям" if mode == "global" else "перемешивание внутри каждой ночи")
        ax.set_ylabel("падение AUC (LOSO)")
    fig.tight_layout()
    fig.savefig(OUT / "C_permutation.png", dpi=130)
    plt.close(fig)
    print("D. SHAP...", flush=True)
    shap_analysis(data, models_by_kind)
    for s in ["46343", "1066528", "7749105"]:
        shap_over_night(data, models_by_kind["lgbm"], s)
    print(shap_three_class(data).round(3))


if __name__ == "__main__":
    main()
