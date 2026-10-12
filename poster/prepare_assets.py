"""Картинки для плаката: обрезанные скриншоты результатов, график экспериментов, формулы. Ничего не обучает.

    python poster/prepare_assets.py <папка клона репозитория>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
RUN = REPO / "results" / "run_2026-10-11"
ACCENT = "#434B9A"


def crop_white(src, dst, pad=-4):
    """Скриншот из отчёта: тёмная полоса с подписью и тёмные поля -> только белая область рисунка."""
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(int)
    white = a.min(axis=2) > 225
    rows = np.where(white.mean(1) > 0.35)[0]
    cols = np.where(white.mean(0) > 0.35)[0]
    box = (max(cols[0] - pad, 0), max(rows[0] - pad, 0), min(cols[-1] + pad, im.width), min(rows[-1] + pad, im.height))
    im.crop(box).save(dst)
    return dst


def experiments_chart():
    exp = pd.read_csv(RUN / "feature_experiments.csv")
    base = exp.iloc[0]
    best = exp.kappa.idxmax()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, gridspec_kw=dict(wspace=0.05))
    y = np.arange(len(exp))[::-1]
    for ax, col, label in [(axes[0], "kappa", "Cohen κ"), (axes[1], "roc_auc", "ROC AUC")]:
        better = exp[col] > base[col] + 1e-9
        colors = ["#0f766e" if i == best else (ACCENT if b else "#9aa0b5") for i, b in zip(exp.index, better)]
        ax.hlines(y, base[col], exp[col], color=colors, lw=2)
        ax.scatter(exp[col], y, color=colors, s=110, zorder=3)
        ax.axvline(base[col], color="#d3494e", ls="--", lw=1.2)
        for yi, v, c in zip(y, exp[col], colors):
            ax.text(v, yi + 0.33, f"{v:.3f}", ha="center", fontsize=14, color=c, fontweight="bold")
        ax.set_xlabel(label, fontsize=16)
        ax.margins(x=0.12)
        ax.grid(axis="x", alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, exp.experiment, fontsize=15)
    axes[0].text(base.kappa, len(exp) - 0.2, "исходная модель", color="#d3494e", fontsize=14, ha="center")
    fig.savefig(ASSETS / "experiments.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


FORMULAS = {
    "logreg": r"$P(\mathrm{sleep}\mid x)=\sigma(w^{\top}x+b),\quad \sigma(z)=\dfrac{1}{1+e^{-z}}$",
    "kappa": r"$\kappa=\dfrac{p_o-p_e}{1-p_e}$",
    "f1": r"$F_1=\dfrac{2\cdot \mathrm{precision}\cdot \mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}$",
    "shapley": r"$\phi_i=\sum_{S\subseteq N\backslash\{i\}}\dfrac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\left[v(S\cup\{i\})-v(S)\right]$",
    "shap": r"$P(\mathrm{sleep}\mid x)=\mathbb{E}\,[f]+\phi_{\mathrm{motion}}+\phi_{\mathrm{hr}}+\phi_{\mathrm{clock}}$",
    "clock": r"$\mathrm{clock}(t)=\dfrac{x(t)-x(t_0)}{\min_t\, x(t)-x(t_0)}$",
    "fpeak": r"$f_{\mathrm{peak}}=\ln\left(1+\dfrac{\max_{k\in C}\, m_k}{\tau}\right)$",
    "factive": r"$f_{\mathrm{active}}=\sum_k A_k\,\mathbf{1}\{m_k\geq\tau\}\,/\,\sum_k A_k$",
    "fhr": r"$f_{\mathrm{HR}}=\dfrac{\bar h_{W_t}-h_{\mathrm{med}}}{h_{\mathrm{med}}}$",
    "motion5": r"$m=\sqrt{v_x+v_y+v_z}$",
}
LATEX = {  # то же в синтаксисе LaTeX — для вставки в PowerPoint (Вставка → Уравнение → LaTeX)
    "logreg": r"P(\mathrm{sleep}\mid x)=\sigma(w^\top x+b),\quad \sigma(z)=\frac{1}{1+e^{-z}}",
    "kappa": r"\kappa=\frac{p_o-p_e}{1-p_e}",
    "f1": r"F_1=\frac{2\cdot \mathrm{precision}\cdot \mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}",
    "shapley": r"\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(|N|-|S|-1)!}{|N|!}\left[v(S\cup\{i\})-v(S)\right]",
    "shap": r"P(\mathrm{sleep}\mid x)=\mathbb{E}[f]+\phi_{\mathrm{motion}}+\phi_{\mathrm{hr}}+\phi_{\mathrm{clock}}",
    "clock": r"\mathrm{clock}(t)=\frac{x(t)-x(t_0)}{\min_t x(t)-x(t_0)}",
    "fpeak": r"f_{\mathrm{peak}}=\ln\left(1+\frac{\max_{k\in C} m_k}{\tau}\right)",
    "factive": r"f_{\mathrm{active}}=\frac{\sum_{k=1}^{12} A_k\,\mathbf{1}\{m_k\ge\tau\}}{\sum_{k=1}^{12} A_k}",
    "fhr": r"f_{\mathrm{HR}}=\frac{\bar h_{W_t}-h_{\mathrm{med}}}{h_{\mathrm{med}}}",
    "motion5": r"m=\sqrt{v_x+v_y+v_z}",
}


def formulas():
    plt.rcParams["mathtext.fontset"] = "cm"
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=40)
        fig.savefig(ASSETS / f"formula_{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
        plt.close(fig)
    (ASSETS / "formulas_latex.txt").write_text("\n".join(f"{k}: {v}" for k, v in LATEX.items()) + "\n",
                                               encoding="utf-8")


plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 13})
MODELS = ["Логистическая регрессия", "k ближайших соседей", "Случайный лес", "Нейросеть (MLP)", "CatBoost"]
MODEL_COLORS = ["#4c72b0", "#8c8c8c", "#55a868", "#c44e52", "#dd8452"]
FEATURE_COLORS = {"motion": "#3b5b92", "hr": "#f0a202", "clock": "#d3494e"}
# значения Шепли из прогона run_all.py (results/run_2026-10-11/RESULTS.md, «Значения Шепли по метрикам»)
SHAPLEY = {  # модель: {признак: (AUC, AP wake, κ, F1 wake)}
    "Логистическая регрессия": {"motion": (0.162, 0.225, 0.185, 0.172), "hr": (0.102, 0.054, 0.073, 0.069),
                                "clock": (0.096, 0.116, 0.134, 0.125)},
    "k ближайших соседей": {"motion": (0.154, 0.187, 0.182, 0.170), "hr": (0.098, 0.061, 0.067, 0.063),
                            "clock": (0.115, 0.170, 0.174, 0.161)},
    "Случайный лес": {"motion": (0.184, 0.231, 0.188, 0.175), "hr": (0.092, 0.042, 0.071, 0.067),
                      "clock": (0.093, 0.206, 0.159, 0.148)},
    "Нейросеть (MLP)": {"motion": (0.184, 0.231, 0.188, 0.175), "hr": (0.112, 0.060, 0.082, 0.076),
                        "clock": (0.082, 0.184, 0.160, 0.148)},
    "CatBoost": {"motion": (0.185, 0.234, 0.192, 0.178), "hr": (0.099, 0.059, 0.077, 0.072),
                 "clock": (0.096, 0.170, 0.163, 0.151)},
}


def shapley_chart():
    feats = ["motion", "hr", "clock"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    width = 0.16
    for i, (m, c) in enumerate(zip(MODELS, MODEL_COLORS)):
        vals = [SHAPLEY[m][f][0] for f in feats]
        ax.bar(np.arange(3) + (i - 2) * width, vals, width, color=c, label=m)
    ax.set_xticks(range(3), feats, fontsize=16)
    ax.set_ylabel("вклад в AUC (над 0.5)")
    ax.set_title("Вклад по Шепли в AUC, все модели", fontweight="bold")
    ax.legend(fontsize=11, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.3)
    ax = axes[1]
    metrics = ["AUC", "AP wake", "κ", "F1 wake"]
    left = np.zeros(4)
    for f in feats:
        share = np.array([np.mean([SHAPLEY[m][f][k] / sum(SHAPLEY[m][g][k] for g in feats) for m in MODELS])
                          for k in range(4)]) * 100
        ax.barh(range(4), share, left=left, color=FEATURE_COLORS[f], label=f)
        for y, (l, s) in enumerate(zip(left, share)):
            ax.text(l + s / 2, y, f"{s:.0f}%", ha="center", va="center", color="white", fontsize=15, fontweight="bold")
        left += share
    ax.set_yticks(range(4), metrics, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("доля прироста качества, % (среднее по 5 моделям)")
    ax.set_title("Чья заслуга: доли признаков", fontweight="bold")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "shapley_poster.png", dpi=220)
    plt.close(fig)


def beeswarm_chart():
    z = np.load(REPO_LOCAL / "outputs" / "repro" / "importance" / "D_shap_rf.npz")
    values, X = z["values"], z["X"]
    feats = ["motion", "hr", "clock"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for j in range(3):
        counts, edges = np.histogram(values[:, j], bins=80)
        dens = counts[np.clip(np.digitize(values[:, j], edges[1:-1]), 0, 79)] / counts.max()
        low, high = np.quantile(X[:, j], [0.05, 0.95])
        color = np.clip((X[:, j] - low) / (high - low), 0, 1)
        order = rng.permutation(len(values))
        sc = ax.scatter(values[order, j], j + (rng.uniform(-1, 1, len(values)) * 0.42 * dens)[order], c=color[order],
                        cmap="coolwarm", s=7, alpha=.8, vmin=0, vmax=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(3), feats, fontsize=17)
    ax.set_ylim(2.6, -0.65)
    ax.set_xlabel("SHAP: на сколько признак сдвинул P(сон) в эпохе", fontsize=15)
    ax.text(-0.45, -0.5, "← к бодрствованию", color="#b2182b", fontsize=15, fontweight="bold")
    ax.text(0.2, -0.5, "к сну →", color="#2166ac", fontsize=15, fontweight="bold")
    notes = [(0.30, 0.42, "покой → сон"), (-0.40, 0.42, "движение → бодрствование"),
             (0.20, 1.45, "ровный пульс → чуть к сну"), (-0.38, 2.45, "начало ночи → бодрствование"),
             (0.22, 2.45, "середина ночи → сон")]
    for x, y, t in notes:
        ax.text(x, y, t, fontsize=14, ha="center", va="center", bbox=dict(fc="white", ec="#bbb", alpha=.9))
    bar = fig.colorbar(sc, ax=ax, ticks=[0, 1], fraction=0.03, pad=0.01)
    bar.ax.set_yticklabels(["мало", "много"], fontsize=13)
    bar.set_label("значение признака", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "beeswarm_poster.png", dpi=220)
    plt.close(fig)


def night_chart(subject="46343"):
    sys.path.insert(0, str(REPO_LOCAL / "repro"))
    import sleeplib as sl
    data = sl.load_paper_features([subject])[subject]
    t = np.arange(len(data["stage"])) * 0.5 / 60
    hr = np.loadtxt(REPO_LOCAL / "data" / "heart_rate" / f"{subject}_heartrate.txt", delimiter=",")
    hr = hr[(hr[:, 0] >= 0) & (hr[:, 0] <= t[-1] * 3600 + 60)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.25]))
    axes[0].fill_between(t, data["motion"], color="#3b5b92", lw=0)
    axes[0].set_ylabel("движение,\ncounts", fontsize=16)
    axes[0].set_title("Движение: activity counts из ускорения часов (сглажено)", fontsize=16, loc="left")
    axes[1].plot(hr[:, 0] / 3600, hr[:, 1], color="#d3494e", lw=1)
    axes[1].set_ylabel("пульс,\nуд/мин", fontsize=16)
    axes[1].set_title("Пульс с оптического датчика (PPG) Apple Watch", fontsize=16, loc="left")
    order = {0: 4, 5: 3, 1: 2, 2: 1, 3: 0, 4: 0}
    axes[2].step(t, [order[int(v)] for v in data["stage"]], where="post", color="k", lw=1.2)
    axes[2].set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "Wake"], fontsize=14)
    axes[2].set_xlabel("часы от начала записи PSG", fontsize=16)
    axes[2].set_ylabel("стадия", fontsize=16)
    axes[2].set_title("Стадии сна по PSG — правильный ответ", fontsize=16, loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(ASSETS / "night_poster.png", dpi=220)
    plt.close(fig)


REPO_LOCAL = HERE.parent  # локальный проект: данные и сохранённые SHAP (их нет в клоне репозитория)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    shapley_chart()
    beeswarm_chart()
    night_chart()
    shots = RUN / "screenshots"
    for name in ["fig2_roc_sleep_wake", "fig_f1_threshold_models", "fig_shapley_auc", "fig_shap_beeswarm_models",
                 "fig_shapley_share", "fig_summary_models", "fig_new_metrics", "fig4_three_class"]:
        crop_white(shots / f"{name}.png", ASSETS / f"{name}.png")
    Image.open(REPO / "results" / "dataset" / "subject_46343_night.png").save(ASSETS / "night.png")
    Image.open(REPO / "docs" / "img" / "clock_explainer.png").save(ASSETS / "clock_explainer.png")
    experiments_chart()
    formulas()
    print(sorted(p.name for p in ASSETS.iterdir()))


if __name__ == "__main__":
    main()
