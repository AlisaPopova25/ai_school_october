"""Разведочный анализ датасета sleep-accel (Apple Watch + PSG).

Сохраняет в outputs/repro/dataset/:
  dataset_summary.csv     — сводка по каждому испытуемому
  coverage.png            — какие сигналы есть на каком интервале времени (относительно начала PSG)
  stage_distribution.png  — доля стадий сна по испытуемым
  subject_<id>_night.png  — сырые сигналы одной ночи + гипнограмма
  subject_<id>_week.png   — шаги и ЧСС за всю амбулаторную неделю
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "repro" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = {-1: "unscored", 0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N4", 5: "REM"}
STAGE_COLORS = {"Wake": "#d62728", "N1": "#9ecae1", "N2": "#4292c6", "N3": "#08519c", "N4": "#08306b",
                "REM": "#2ca02c", "unscored": "#bbbbbb"}


def load(subject):
    motion = pd.read_csv(DATA / f"motion/{subject}_acceleration.txt", sep=" ", header=None,
                         names=["t", "x", "y", "z"], on_bad_lines="skip").dropna()
    hr = pd.read_csv(DATA / f"heart_rate/{subject}_heartrate.txt", header=None, names=["t", "bpm"])
    labels = pd.read_csv(DATA / f"labels/{subject}_labeled_sleep.txt", sep=" ", header=None, names=["t", "stage"])
    steps = pd.read_csv(DATA / f"steps/{subject}_steps.txt", header=None, names=["t", "steps"])
    return motion, hr, labels, steps


def summarize(subject, motion, hr, labels, steps):
    night_m = motion[(motion.t >= 0) & (motion.t <= labels.t.max() + 30)]
    night_hr = hr[(hr.t >= 0) & (hr.t <= labels.t.max() + 30)]
    dt_m = np.diff(night_m.t.values)
    dt_hr = np.diff(night_hr.t.values)
    scored = labels[labels.stage >= 0]
    row = dict(
        subject=subject,
        psg_hours=(labels.t.max() + 30) / 3600,
        epochs_total=len(labels),
        epochs_scored=len(scored),
        motion_rows=len(motion),
        motion_start_h=motion.t.min() / 3600,
        motion_end_h=motion.t.max() / 3600,
        motion_hz_night=1 / np.median(dt_m) if len(dt_m) else np.nan,
        motion_gaps_gt1s_night=int((dt_m > 1).sum()),
        hr_rows=len(hr),
        hr_start_days=hr.t.min() / 86400,
        hr_median_interval_s_night=np.median(dt_hr) if len(dt_hr) else np.nan,
        hr_mean_bpm_night=night_hr.bpm.mean(),
        steps_start_days=steps.t.min() / 86400,
        steps_end_days=steps.t.max() / 86400,
        steps_interval_s=np.median(np.diff(steps.t.values)),
    )
    for code, name in STAGE_NAMES.items():
        row[f"min_{name}"] = (labels.stage == code).sum() * 0.5
    return row


def plot_night(subject, motion, hr, labels):
    end = labels.t.max() + 30
    m = motion[(motion.t >= 0) & (motion.t <= end)]
    h = hr[(hr.t >= 0) & (hr.t <= end)]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1.4, 1.2]))
    for c, col in zip("xyz", ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        axes[0].plot(m.t / 3600, m[c], lw=0.3, color=col, label=c)
    axes[0].set_ylabel("ускорение, g")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].set_title(f"Испытуемый {subject}: ночь PSG (t=0 — начало записи PSG)")
    mag = np.sqrt(m.x ** 2 + m.y ** 2 + m.z ** 2)
    enmo = np.clip(mag - 1, 0, None)
    axes[1].plot(m.t / 3600, enmo, lw=0.3, color="k")
    axes[1].set_ylabel("ENMO = max(|a|-1,0)")
    axes[2].plot(h.t / 3600, h.bpm, ".-", ms=2, lw=0.5, color="#d62728")
    axes[2].set_ylabel("ЧСС, уд/мин")
    order = {0: 4, 5: 3, 1: 2, 2: 1, 3: 0, 4: 0, -1: np.nan}
    axes[3].step(labels.t / 3600, labels.stage.map(order), where="post", color="k", lw=0.8)
    axes[3].set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "Wake"])
    axes[3].set_ylabel("стадия (PSG)")
    axes[3].set_xlabel("часы от начала PSG")
    fig.tight_layout()
    fig.savefig(OUT / f"subject_{subject}_night.png", dpi=130)
    plt.close(fig)


def plot_week(subject, hr, steps, labels):
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    axes[0].bar(steps.t / 86400, steps.steps, width=600 / 86400, color="#555")
    axes[0].set_ylabel("шаги / интервал")
    axes[0].set_title(f"Испытуемый {subject}: амбулаторный период (дни относительно начала PSG)")
    axes[1].plot(hr.t / 86400, hr.bpm, lw=0.3, color="#d62728")
    axes[1].set_ylabel("ЧСС, уд/мин")
    axes[1].set_xlabel("дни от начала PSG")
    for ax in axes:
        ax.axvspan(0, (labels.t.max() + 30) / 86400, color="#2ca02c", alpha=0.2, label="ночь PSG")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"subject_{subject}_week.png", dpi=130)
    plt.close(fig)


def plot_coverage(summary):
    s = summary.sort_values("steps_start_days")
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, r in enumerate(s.itertuples()):
        ax.plot([r.steps_start_days, r.steps_end_days], [i + 0.27] * 2, lw=3, color="#7f7f7f", label="steps" if i == 0 else None)
        ax.plot([r.hr_start_days, r.psg_hours / 24], [i + 0.09] * 2, lw=3, color="#d62728", label="heart rate" if i == 0 else None)
        ax.plot([r.motion_start_h / 24, r.motion_end_h / 24], [i - 0.09] * 2, lw=3, color="#1f77b4", label="motion" if i == 0 else None)
        ax.plot([0, r.psg_hours / 24], [i - 0.27] * 2, lw=3, color="#2ca02c", label="PSG labels" if i == 0 else None)
    ax.set_yticks(range(len(s)), s.subject)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("дни относительно начала PSG")
    ax.set_title("Покрытие сигналов по испытуемым")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png", dpi=130)
    plt.close(fig)


def plot_stages(summary):
    names = ["Wake", "N1", "N2", "N3", "REM"]
    s = summary.copy()
    s["min_N3"] = s["min_N3"] + s["min_N4"]
    scored = s[[f"min_{n}" for n in names]].sum(axis=1)
    frac = s[[f"min_{n}" for n in names]].div(scored, axis=0)
    frac.index = s.subject
    frac = frac.sort_values("min_Wake")
    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(frac))
    for n in names:
        ax.bar(frac.index.astype(str), frac[f"min_{n}"], bottom=bottom, color=STAGE_COLORS[n], label=n)
        bottom += frac[f"min_{n}"].values
    ax.set_ylabel("доля размеченных эпох")
    ax.set_title("Распределение стадий сна (PSG) по испытуемым")
    ax.tick_params(axis="x", rotation=70)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(OUT / "stage_distribution.png", dpi=130)
    plt.close(fig)


def main():
    subjects = sorted(p.name.split("_")[0] for p in (DATA / "labels").glob("*_labeled_sleep.txt"))
    rows = []
    for subject in subjects:
        print("loading", subject, flush=True)
        motion, hr, labels, steps = load(subject)
        rows.append(summarize(subject, motion, hr, labels, steps))
        if subject in ("46343", "8686948"):
            plot_night(subject, motion, hr, labels)
            plot_week(subject, hr, steps, labels)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "dataset_summary.csv", index=False, float_format="%.3f")
    plot_coverage(summary)
    plot_stages(summary)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summary.describe().T[["mean", "std", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
