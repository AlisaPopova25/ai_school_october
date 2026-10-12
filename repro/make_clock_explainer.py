"""Картинка «откуда берётся clock»: шаги за прошлые дни -> придуманный свет -> модель внутренних часов -> clock ночью.

    python repro/make_clock_explainer.py [subject]

Ничего не обучает: повторяет расчёт source/preprocessing/time/clock_proxy/circadian_model.py для одного человека,
но сохраняет весь ход модели, а не только 9 часов ночи. Результат: outputs/repro/shapley_guide/clock_explainer.png.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from source.preprocessing.time.clock_proxy import circadian_model as cm  # noqa: E402


def run(subject):
    """Те же шаги, что cm.build_for_subject, но возвращает весь ход модели."""
    steps = cm.clean_steps(pd.read_csv(ROOT / f"data/steps/{subject}_steps.txt", header=None).values.astype(float))
    steps = steps[np.argsort(steps[:, 0])]
    score = np.loadtxt(ROOT / f"outputs/cropped/{subject}_cleaned_psg.out")
    dt = cm.BIN_MINUTES * cm.SECONDS_PER_MINUTE
    duration_days = min((steps[:, 0].max() - steps[:, 0].min()) / cm.SECONDS_PER_DAY, cm.MAX_DAYS_TO_AVERAGE)
    bins_per_day = cm.SECONDS_PER_DAY // dt
    cropped_start = steps[:, 0].max() - cm.SECONDS_PER_DAY * math.floor(duration_days)
    end_light_window = steps[:, 0].max() - dt / 2
    end_timestamp = steps[:, 0].max() + 3 * cm.SECONDS_PER_DAY
    if score[:, 0].max() < steps[:, 0].max():
        end_timestamp = score[:, 0].max() + 3 * cm.SECONDS_PER_DAY
        end_light_window = score[:, 0].max() - dt / 2
        cropped_start = score[:, 0].max() - cm.SECONDS_PER_DAY * math.floor(duration_days)
    total, count = np.zeros(bins_per_day), np.zeros(bins_per_day)
    for i, t in enumerate(np.arange(cropped_start + dt / 2, end_light_window + 1e-9, dt)):
        in_bin = (steps[:, 0] >= t - dt / 2) & (steps[:, 0] < t + dt / 2)
        total[i % bins_per_day] += steps[in_bin, 1].sum()
        count[i % bins_per_day] += 1
    average_steps = total / np.maximum(count, 1)
    lux_day = cm.steps_to_light(average_steps, cm.local_hour(cropped_start), dt)
    lux = np.tile(lux_day, cm.NUM_DAYS_TO_SIMULATE)
    timestamps = end_timestamp - dt * np.arange(cm.NUM_DAYS_TO_SIMULATE * bins_per_day)[::-1]
    psg_start = score[:, 0].min()
    lux[timestamps > psg_start] = 0
    keep = timestamps <= psg_start + cm.HOURS_OVERHANG_TO_CROP * cm.SECONDS_PER_HOUR
    timestamps, lux = timestamps[keep], lux[keep]
    hours = (timestamps - timestamps.min()) / cm.SECONDS_PER_HOUR
    tc, y = cm.simulate(hours, cm.lux_to_alpha(lux), hours.max())
    t_rel = (tc * cm.SECONDS_PER_HOUR + timestamps.min() - psg_start) / cm.SECONDS_PER_HOUR  # часы от начала PSG
    lux_t = (timestamps - psg_start) / cm.SECONDS_PER_HOUR
    return t_rel, y[:, 0], lux_t, lux, average_steps, score


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "4426783"
    t, x, lux_t, lux, average_steps, score = run(subject)
    night_end = (score[:, 0].max() - score[:, 0].min()) / 3600
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw=dict(height_ratios=[1, 1.6, 1.3]))

    ax = axes[0]  # средний день шагов и придуманный свет
    hours = (cm.PSG_START_CLOCK_HOUR + np.arange(len(average_steps)) * cm.BIN_MINUTES / 60) % 24
    order = np.argsort(hours)
    ax.bar(hours[order], average_steps[order], width=cm.BIN_MINUTES / 60, color="#8c8c8c", label="шагов за 10 мин")
    ax.axhline(20, color="k", ls=":", lw=1)
    ax.text(0.1, 22, "порог 20 шагов: выше — «человек на свету»", fontsize=9)
    ax2 = ax.twinx()
    day_lux = cm.steps_to_light(average_steps, cm.PSG_START_CLOCK_HOUR, cm.BIN_MINUTES * 60)
    ax2.step(hours[order], day_lux[order], where="mid", color="#f0a202", lw=1.5, label="придуманный свет, люкс")
    ax2.set_ylabel("свет, люкс", color="#b07000")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("время суток, ч")
    ax.set_ylabel("шагов за 10 мин")
    ax.set_title(f"Шаг 1. Средний день по шагам с часов (испытуемый {subject}, до 5 дней) → свет: 1000 лк днём, "
                 "500 утром/вечером, 50 ночью, 0 — если не ходит", fontsize=11, fontweight="bold")

    ax = axes[1]  # несколько последних суток модели
    show = t > -72
    ax.plot(t[show], x[show], color="#3b5b92", lw=2, label="x — состояние внутренних часов (без единиц)")
    lit = (lux_t > -72) & (lux > 0)
    colors = {50: "#fde9a9", 500: "#f6c244", 1000: "#e08a00"}
    ax.bar(lux_t[lit], 0.15, bottom=-1.6, width=cm.BIN_MINUTES / 60, align="edge",
           color=[colors.get(int(v), "#f0a202") for v in lux[lit]])
    ax.text(-72, -1.4, "свет (по шагам): светлее — 50 лк, темнее — 1000 лк; пусто — темно", fontsize=9)
    ax.axvspan(0, night_end, color="#d3494e", alpha=.15, label="ночь в лаборатории (PSG): свет = 0")
    for day in range(-3, 1):
        ax.axvline(day * 24, color="#bbb", lw=.8)
        ax.text(day * 24 + 0.3, -1.25, "23:00", fontsize=8, color="#777")
    i_min = np.argmin(np.where((t > 0) & (t < 12), x, np.inf))
    ax.scatter([t[i_min]], [x[i_min]], color="k", zorder=5)
    ax.annotate(f"дно волны x (минимум) ≈ {(cm.PSG_START_CLOCK_HOUR + t[i_min]) % 24:.1f} ч ночи →\n"
                "здесь clock = 1 (максимум clock)", (t[i_min], x[i_min]),
                (t[i_min] - 30, x[i_min] - 0.1), arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.set_ylim(-1.65, 1.5)
    ax.set_xlabel("часы от начала ночи в лаборатории (0 = 23:00)")
    ax.set_title("Шаг 2. Модель внутренних часов (Forger 1999): волна с периодом ~24 ч, свет её подталкивает",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(-1.65, 1.75)
    ax.grid(alpha=.3)

    ax = axes[2]  # окно ночи и нормировка
    night = (t >= 0) & (t <= night_end)
    x0, xmin = np.interp(0, t, x), x[night].min()
    clock = (x[night] - x0) / (xmin - x0)
    ax.plot(t[night], clock, color="#d3494e", lw=2.5, label="clock = (x − x в начале ночи) / (минимум x − x в начале)")
    ax.axhline(0, color="k", lw=.6)
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.text(0.05, 0.04, "0 = лёг спать", fontsize=10)
    ax.text(t[night][np.argmax(clock)], 1.04, "1 = максимум clock = дно (минимум) волны x: шкала перевёрнута",
            ha="center", fontsize=10)
    ax.set_ylim(-0.25, 1.2)
    ax.set_xlabel("часы от начала ночи")
    ax.set_ylabel("clock")
    ax.set_title("Шаг 3. Из волны берём только кусок ночи и растягиваем его в шкалу от 0 до 1 — это и есть признак clock",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    out = ROOT / "outputs" / "repro" / "shapley_guide" / "clock_explainer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("записано:", out.relative_to(ROOT))
    print(f"x: от {x[show].min():.2f} до {x[show].max():.2f}; минимум в ночи через {t[i_min]:.2f} ч; "
          f"свет есть в {np.mean(lux[(lux_t > -24) & (lux_t < 0)] > 0) * 100:.0f}% интервалов последних суток")


if __name__ == "__main__":
    main()
