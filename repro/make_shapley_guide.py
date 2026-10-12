"""Картинки-пояснения к графикам Шепли и SHAP: те же графики с пронумерованными подписями «что здесь видно».

    python repro/make_shapley_guide.py

Ничего не обучает: берёт сохранённые результаты repro/03_feature_importance.py (outputs/repro/importance/:
D_shap_*.npz, A_subsets.csv, A_shapley_auc.csv и готовые рисунки) и пишет outputs/repro/shapley_guide/guide_*.png.
Пояснения к картинкам — docs/SHAPLEY_RU.md.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "repro" / "importance"
OUT = ROOT / "outputs" / "repro" / "shapley_guide"
FEATURES = ["motion", "hr", "clock"]
MARK = dict(boxstyle="circle,pad=0.25", fc="#ffd400", ec="k", lw=1.2)


def mark(ax, n, xy, text_xy=None, coords="data"):
    """Жёлтый кружок с номером; если задан text_xy — со стрелкой к точке xy."""
    if text_xy is None:
        ax.annotate(str(n), xy, xycoords=coords, ha="center", va="center", fontsize=12, fontweight="bold", bbox=MARK,
                    zorder=10)
    else:
        ax.annotate(str(n), xy, text_xy, xycoords=coords, textcoords=coords, ha="center", va="center", fontsize=12,
                    fontweight="bold", bbox=MARK, zorder=10, arrowprops=dict(arrowstyle="->", lw=1.8, color="k"))


def legend_panel(ax, items, title):
    ax.axis("off")
    ax.text(0, 1, title, fontsize=13, fontweight="bold", va="top", transform=ax.transAxes)
    top = 0.97 - 0.045 * (title.count("\n") + 1)
    lines = [text.count("\n") + 1 for _, text in items]
    step = min(0.045, top / (sum(lines) + 0.9 * len(items)))  # шаг на строку: всё помещается в панель
    y = top
    for (n, text), k in zip(items, lines):
        ax.text(0.0, y, str(n), fontsize=11, fontweight="bold", va="top", ha="center", bbox=MARK,
                transform=ax.transAxes)
        ax.text(0.07, y + 0.012, text, fontsize=10.5, va="top", transform=ax.transAxes)
        y -= step * (k + 0.9)


def load(kind):
    z = np.load(SRC / f"D_shap_{kind}.npz")
    return z["values"], z["X"]


def swarm(ax, values, X, rng):
    for j in range(3):
        counts, edges = np.histogram(values[:, j], bins=80)
        dens = counts[np.clip(np.digitize(values[:, j], edges[1:-1]), 0, 79)] / counts.max()
        low, high = np.quantile(X[:, j], [0.05, 0.95])
        color = np.clip((X[:, j] - low) / (high - low), 0, 1)
        order = rng.permutation(len(values))
        ax.scatter(values[order, j], j + (rng.uniform(-1, 1, len(values)) * 0.4 * dens)[order], c=color[order],
                   cmap="coolwarm", s=4, alpha=.75, vmin=0, vmax=1)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(range(3), FEATURES, fontsize=12)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("SHAP: на сколько признак сдвинул P(сон) в этой эпохе", fontsize=11)
    ax.grid(axis="x", alpha=.3)


# --------------------------------------------------------------------------- 1. бисворм
def guide_beeswarm():
    values, X = load("rf")
    fig = plt.figure(figsize=(17, 6.6))
    ax = fig.add_axes([0.06, 0.12, 0.5, 0.78])
    swarm(ax, values, X, np.random.default_rng(0))
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    bar = fig.colorbar(sm, ax=ax, ticks=[0, 1], fraction=0.03, pad=0.01)
    bar.ax.set_yticklabels(["мало", "много"])
    bar.set_label("значение признака в эпохе")
    ax.set_title("Случайный лес, сон/бодрствование: SHAP каждой эпохи", fontsize=13, fontweight="bold")
    ax.text(0.25, -0.45, "← к бодрствованию", fontsize=11, color="#b2182b", transform=ax.get_yaxis_transform())
    ax.text(0.62, -0.45, "к сну →", fontsize=11, color="#2166ac", transform=ax.get_yaxis_transform())
    # одна конкретная точка
    i = int(np.argmin(np.abs(values[:, 0] + 0.45) + 10 * (X[:, 0] < 15)))
    ax.scatter([values[i, 0]], [0], s=140, facecolor="none", edgecolor="k", lw=2, zorder=9)
    mark(ax, 1, (values[i, 0], 0), (values[i, 0] - 0.04, -0.42))
    mark(ax, 2, (0.27, 0.1), (0.17, 0.55))
    mark(ax, 3, (-0.35, 0.0), (-0.35, 0.5))
    mark(ax, 4, (0.18, 2.0), (0.3, 2.45))
    mark(ax, 5, (-0.42, 2.0), (-0.42, 2.45))
    mark(ax, 6, (0.1, 1.0), (0.3, 1.0))
    mark(ax, 7, (0.33, -0.33), (0.43, -0.33))
    ax.annotate("", (0.33, 0.33), (0.33, -0.33), arrowprops=dict(arrowstyle="<->", lw=1.5))
    legend_panel(fig.add_axes([0.63, 0.05, 0.36, 0.9]), [
        (1, f"Одна точка — одна эпоха (30 с) человека, которого\nмодель не видела. Здесь motion = {X[i, 0]:.0f}"
            f" (движение),\nSHAP = {values[i, 0]:+.2f}: движение снизило P(сон) на {-values[i, 0]:.2f}."),
        (2, "Синий ком: motion = 0 (покой), таких эпох 52%.\nПокой поднимает P(сон) примерно на +0.3."),
        (3, "Красный хвост: сильное движение. Эпох мало (≈1%),\nно сдвиг огромный — до −0.55 к бодрствованию."),
        (4, "clock, красное справа: середина/конец ночи → к сну."),
        (5, "clock, синее слева: начало ночи (засыпание)\n→ к бодрствованию."),
        (6, "hr: синие (ровный пульс) — немного к сну, красные\n(неровный) — к бодрствованию. Рой узкий: hr двигает\n"
            "P(сон) слабее, чем motion и clock."),
        (7, "Толщина роя по вертикали = сколько эпох с таким\nSHAP. Высота точки внутри строки ничего не значит."),
    ], "Как читать")
    return save(fig, "guide_1_beeswarm.png")


# --------------------------------------------------------------------------- 2. графики зависимости
def dependence(ax, values, X, j, color_j, title):
    low, high = np.quantile(X[:, color_j], [0.05, 0.95])
    sc = ax.scatter(X[:, j], values[:, j], c=X[:, color_j], cmap="coolwarm", s=5, alpha=.7, vmin=low, vmax=high)
    ax.axhline(0, color="k", lw=.8)
    if FEATURES[j] == "motion":
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(f"значение {FEATURES[j]}", fontsize=11)
    ax.set_ylabel(f"SHAP({FEATURES[j]})", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(sc, ax=ax, label=f"цвет = {FEATURES[color_j]}")
    ax.grid(alpha=.3)


def guide_dependence():
    rf, Xrf = load("rf")
    mlp, Xmlp = load("mlp")
    fig = plt.figure(figsize=(18, 11))
    a1 = fig.add_axes([0.05, 0.56, 0.27, 0.38])
    a2 = fig.add_axes([0.38, 0.56, 0.27, 0.38])
    a3 = fig.add_axes([0.05, 0.07, 0.27, 0.38])
    a4 = fig.add_axes([0.38, 0.07, 0.27, 0.38])
    dependence(a1, rf, Xrf, 0, 2, "Случайный лес: motion")
    dependence(a2, mlp, Xmlp, 0, 2, "Нейросеть: motion")
    dependence(a3, rf, Xrf, 2, 0, "Случайный лес: clock")
    dependence(a4, rf, Xrf, 1, 2, "Случайный лес: hr")
    mark(a1, 1, (0, 0.15), (0.35, 0.33))
    mark(a1, 2, (2.5, -0.1), (8, 0.15))
    mark(a1, 3, (60, -0.48), (25, -0.2))
    mark(a2, 4, (8, -0.25), (2, -0.55))
    mark(a3, 5, (0.05, -0.4), (0.25, -0.4))
    mark(a3, 6, (0.5, 0.05), (0.42, 0.28))
    mark(a3, 7, (0.82, -0.25), (0.65, -0.38))
    mark(a4, 8, (0.15, 0.15), (0.6, 0.3))
    mark(a4, 9, (2.6, -0.2), (2.6, 0.1))
    legend_panel(fig.add_axes([0.71, 0.04, 0.28, 0.92]), [
        (1, "Вертикальная линия при motion = 0: половина эпох\nс одним и тем же значением. SHAP у них разный\n"
            "(от −0.05 до +0.4): вклад motion зависит от других\nпризнаков — это взаимодействие."),
        (2, "Наклон вниз: чем больше движения, тем сильнее\nдовод за бодрствование."),
        (3, "Справа точек мало: сильное движение редкое.\nРедко, но вклад самый большой."),
        (4, "Нейросеть — гладкая кривая, у леса — ступеньки\nи полосы: дерево режет ось порогами, между\n"
            "порогами предсказание не меняется."),
        (5, "Начало ночи (clock ≈ 0): сильно к бодрствованию —\nчеловек ещё засыпает."),
        (6, "«Нити»: соседние эпохи одной ночи почти не\nотличаются по clock (шаг 0.0016) и ложатся\nцепочкой."),
        (7, "Провал около 0.8: несколько ночей с пробуждениями\nв это время; лес это запомнил (переобучение)."),
        (8, "hr: плотное облако 0–1 (90% эпох). Ровный пульс\n(hr мал) — слегка к сну."),
        (9, "Большая вариабельность пульса — редкие точки,\nк бодрствованию (пробуждения, артефакты)."),
    ], "Ось x — значение признака, ось y — его SHAP,\nцвет — ДРУГОЙ признак (шкала справа)")
    return save(fig, "guide_2_dependence.png")


# --------------------------------------------------------------------------- 3. как считается Шепли по качеству
def guide_shapley():
    sub = pd.read_csv(SRC / "A_subsets.csv")
    sub = sub[sub.model == "lr"].set_index("subset").auc
    sh = pd.read_csv(SRC / "A_shapley_auc.csv")
    sh = sh[sh.model == "lr"].set_index("feature")
    auc = {frozenset(k.split("+")): v for k, v in sub.items()}
    auc[frozenset()] = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    orders = [("motion", "hr", "clock"), ("motion", "clock", "hr"), ("hr", "motion", "clock"),
              ("clock", "motion", "hr"), ("hr", "clock", "motion"), ("clock", "hr", "motion")]
    colors = {"motion": "#3b5b92", "hr": "#f0a202", "clock": "#d3494e"}
    gains = []
    for i, order in enumerate(orders):
        have, left = frozenset(), 0.5
        for f in order:
            gain = auc[have | {f}] - auc[have]
            ax.barh(i, gain, left=left, color=colors[f], edgecolor="white", height=0.6)
            if gain > 0.03:  # на узком отрезке подпись не помещается
                ax.text(left + gain / 2, i, f"{f}\n+{gain:.3f}", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
            else:
                ax.text(left + gain / 2, i - 0.42, f"{f} +{gain:.3f}", ha="center", va="bottom", fontsize=8)
            if f == "motion":
                gains.append(gain)
            have, left = have | {f}, left + gain
    ax.set_yticks(range(6), [" → ".join(o) for o in orders])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.87)
    ax.axvline(auc[frozenset(FEATURES)], color="k", ls="--", lw=1)
    ax.set_xlabel("AUC (логистическая регрессия, LOSO): признаки добавляются по очереди", fontsize=11)
    ax.set_title("Шаг 1: все 6 порядков добавления признаков", fontsize=13, fontweight="bold")
    mark(ax, 1, (0.5 + gains[0] / 2, 0), (0.5 + gains[0] / 2, -0.62))
    mark(ax, 2, (auc[frozenset(["hr", "clock"])] + gains[4] / 2, 4.28), (0.86, 5.0))
    ax.text(0.505, 6.05, "1 — motion добавлен первым: прирост большой (+%.3f).\n"
                         "2 — motion добавлен последним: прирост маленький (+%.3f) —\n"
                         "часть его информации уже есть в hr и clock." % (gains[0], gains[4]),
            fontsize=10, va="top")
    ax.set_ylim(7.3, -0.9)

    ax = axes[1]
    left = 0.5
    ax.bar(0, 0.5, color="#bbbbbb")
    ax.text(0, 0.51, "случайная\nмодель\n0.500", ha="center", va="bottom", fontsize=10)
    for k, f in enumerate(FEATURES, 1):
        v = sh.loc[f, "shapley_auc"]
        ax.bar(k, v, bottom=left, color=colors[f])
        ax.text(k, left + v + 0.004, f"+{v:.3f}\n({100 * sh.loc[f, 'share']:.0f}%)", ha="center", fontsize=10)
        left += v
    ax.bar(4, left, color="#444444")
    ax.text(4, left + 0.004, f"полная\nмодель\n{left:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(5), ["0.5", "motion", "hr", "clock", "итог"])
    ax.set_ylim(0.45, 0.93)
    ax.set_ylabel("AUC")
    ax.set_title("Шаг 2: значение Шепли = средний прирост по 6 порядкам", fontsize=13, fontweight="bold")
    mark(ax, 3, (1, 0.5 + sh.loc["motion", "shapley_auc"] / 2), (1.55, 0.6))
    ax.text(-0.4, 0.915, "3 — вклад motion = среднее его приростов из 6 строк слева.\n"
                         "Вклады трёх признаков в сумме дают ровно AUC полной модели − 0.5.", fontsize=10, va="top")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return save(fig, "guide_3_shapley_orders.png")


# --------------------------------------------------------------------------- 4. почему рой несимметричен
def guide_asymmetry():
    rf, _ = load("rf")
    mlp, _ = load("mlp")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    for ax, values, name, color in [(axes[0], rf, "Случайный лес (веса классов, средняя P(сон) ≈ 0.6)", "#55a868"),
                                    (axes[1], mlp, "Нейросеть статьи (без весов, средняя P(сон) ≈ 0.9)", "#c44e52")]:
        ax.hist(values[:, 0], bins=np.linspace(-0.9, 0.5, 71), color=color, edgecolor="white")
        ax.axvline(0, color="k", lw=.8)
        top = values[:, 0].max()
        ax.axvline(top, color="k", ls="--", lw=1)
        ax.text(top, ax.get_ylim()[1] * 0.92, f" максимум\n +{top:.2f}", fontsize=10, va="top")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("SHAP(motion)")
        ax.set_yscale("log")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("эпох (логарифмическая шкала)")
    mark(axes[1], 1, (0.05, 300), (0.25, 300))
    mark(axes[1], 2, (-0.6, 3), (-0.6, 60))
    fig.suptitle("P(сон) не может быть больше 1: признак поднимает её не выше чем до 1, а опустить может почти до 0.\n"
                 "1 — у нейросети средняя P(сон) уже ≈ 0.9, вверх двигать почти некуда.   2 — вниз есть куда: до −0.8.",
                 fontsize=12)
    fig.tight_layout()
    return save(fig, "guide_4_asymmetry.png")


# --------------------------------------------------------------------------- 5–6. подписи на готовых рисунках
def annotate_png(name, marks, items, title, out, size, legend_width=0.33):
    img = plt.imread(SRC / name)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1 - legend_width, 1])
    ax.imshow(img)
    ax.axis("off")
    for n, xy, text_xy in marks:
        mark(ax, n, xy, text_xy)
    legend_panel(fig.add_axes([1 - legend_width + 0.02, 0.03, legend_width - 0.03, 0.94]), items, title)
    return save(fig, out)


def guide_three_class():
    return annotate_png("D_shap_three_class.png", [
        (1, (110, 75), (175, 140)), (2, (305, 88), (370, 110)), (3, (490, 168), (528, 300)),
        (4, (760, 115), (860, 60)), (5, (673, 390), (760, 440)), (6, (985, 380), (1050, 300)),
        (7, (1640, 215), (1560, 110)), (8, (1925, 395), (2000, 455)), (9, (2255, 135), (2190, 60)),
    ], [
        (1, "Средний |SHAP| по классам. motion важнее всего\nдля wake: движение = не спит."),
        (2, "hr важнее всего для NREM: ровный пульс —\nглубокий сон."),
        (3, "clock важен для REM: REM мало в начале ночи\nи много под утро."),
        (4, "hr → REM: пик при средней вариабельности\n(0.3–0.8) — в REM пульс неровный."),
        (5, "Очень ровный пульс (hr ≈ 0, глубокий сон)\n— против REM."),
        (6, "Очень большая вариабельность (> 1, это\nбодрствование) — снова против REM. Форма «горка»:\n"
            "линейная модель такое не уловит."),
        (7, "motion → wake: чем больше движения, тем\nсильнее довод за бодрствование (монотонно)."),
        (8, "clock → REM: начало ночи — сильно против REM."),
        (9, "Конец ночи — за REM. Цвет точек во всех\nграфиках = clock (синий — начало ночи)."),
    ], "LightGBM, три класса", "guide_5_three_class.png", (24, 7.2), 0.3)


def guide_over_night():
    return annotate_png("D_shap_over_night_46343.png", [
        (1, (347, 82), (430, 120)), (2, (175, 395), (250, 455)), (3, (1550, 470), (1450, 470)),
        (4, (1550, 540), (1450, 600)), (5, (1550, 770), (1450, 800)), (6, (1004, 610), (1004, 720)),
        (7, (300, 240), (300, 300)),
    ], [
        (1, "Плато hr ≈ +4: очень ровный пульс в глубоком\nсне N3 (внизу на гипнограмме — N3). Плато ровное:\n"
            "признак долго не выходит за пороги деревьев."),
        (2, "clock в начале ночи ≈ −4: «сейчас засыпают»\n→ к бодрствованию; дальше растёт к +1…+2."),
        (3, "motion падает до −6: человек двигается."),
        (4, "Сумма вкладов сильно отрицательна → P(wake)\nпочти 1 (средняя панель)."),
        (5, "Гипнограмма PSG (правда): здесь действительно\nWake — модель угадала."),
        (6, "Пик P(wake) без Wake на гипнограмме — ложная\nтревога (движение во сне)."),
        (7, "Верхняя панель — SHAP каждого признака в\nlog-odds сна. Выше нуля — к сну, ниже —\nк бодрствованию."),
    ], "Одна ночь (испытуемый 46343), LightGBM", "guide_6_over_night.png", (20, 10), 0.32)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    return OUT / name


def main():
    for make in [guide_beeswarm, guide_dependence, guide_shapley, guide_asymmetry, guide_three_class,
                 guide_over_night]:
        print("записано:", make().relative_to(ROOT))


if __name__ == "__main__":
    main()
