"""Расширенные признаки на эпоху (для 04_improve_model.py).

Строятся из тех же обрезанных сигналов (outputs/cropped/), что и признаки статьи, и для тех же валидных эпох,
поэтому их можно напрямую сравнивать с моделью статьи. Кэш: outputs/repro/ext_features/<id>.npz

Группы:
  paper  — motion, hr, clock (как в статье);
  counts — activity counts в окнах ±1…±20 мин, отдельно прошлое/будущее (log1p);
  accel  — из сырого ускорения: ENMO, изменение угла запястья (van Hees), доля «неподвижных» секунд,
           время с последнего / до следующего движения;
  hrlvl  — уровень ЧСС (нормированный на человека) и его разброс в окнах ±2.5…±20 мин, тренд;
  time   — часы от начала записи.
"""
import numpy as np
import pandas as pd

import sleeplib as sl
from source.preprocessing.raw_data_processor import RawDataProcessor

CACHE = sl.OUT / "ext_features"
CACHE.mkdir(parents=True, exist_ok=True)
WINDOWS_MIN = [1, 2.5, 5, 10, 20]


def _per_second(t, values, t0, n):
    """Среднее значение по секундам на сетке t0..t0+n-1 (NaN где нет отсчётов)."""
    idx = np.floor(t - t0).astype(int)
    ok = (idx >= 0) & (idx < n)
    s = np.bincount(idx[ok], weights=values[ok], minlength=n)
    c = np.bincount(idx[ok], minlength=n)
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _fill(x):
    good = ~np.isnan(x)
    if good.all():
        return x
    return np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])


def _window_mean(cs, centers, half):
    """Среднее по окну [center-half, center+half] через кумулятивную сумму cs (cs[0] = 0)."""
    n = len(cs) - 1
    lo = np.clip(centers - half, 0, n)
    hi = np.clip(centers + half + 1, 0, n)
    return (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)


def _window_range(cs, centers, a, b):
    n = len(cs) - 1
    lo = np.clip(centers + a, 0, n)
    hi = np.clip(centers + b, 0, n)
    return (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)


def build_subject(subject, paper):
    path = CACHE / f"{subject}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return dict(X=z["X"], names=list(z["names"]))

    epochs = RawDataProcessor.get_valid_epochs(subject)
    te = np.array([e.timestamp for e in epochs])
    assert len(te) == len(paper["stage"]), subject

    motion = pd.read_csv(sl.CROPPED_DIR / f"{subject}_cleaned_motion.out", sep=" ", header=None).values
    hr = pd.read_csv(sl.CROPPED_DIR / f"{subject}_cleaned_hr.out", sep=" ", header=None).values
    counts = pd.read_csv(sl.CROPPED_DIR / f"{subject}_cleaned_counts.out", header=None).values

    t0 = np.floor(min(motion[:, 0].min(), te.min() - 1200))
    t1 = np.ceil(max(motion[:, 0].max(), te.max() + 1230))
    n = int(t1 - t0) + 1
    centers = (te + 15 - t0).astype(int)

    # --- сырое ускорение: секунда -> ENMO и угол запястья
    ax, ay, az = (_fill(_per_second(motion[:, 0], motion[:, k], t0, n)) for k in (1, 2, 3))
    mag = _per_second(motion[:, 0], np.sqrt((motion[:, 1:4] ** 2).sum(1)), t0, n)
    enmo = np.clip(_fill(mag) - 1, 0, None)
    angle = np.degrees(np.arctan2(az, np.sqrt(ax ** 2 + ay ** 2)))
    angle5 = pd.Series(angle).rolling(5, center=True, min_periods=1).median().values
    dangle = np.abs(np.diff(angle5, prepend=angle5[0]))
    active = (pd.Series(dangle).rolling(5, center=True, min_periods=1).max().values > 5 / 5).astype(float)

    # время с последнего / до следующего движения (сек), обрезано 60 мин
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(active > 0, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(active > 0, idx, 10 ** 9)[::-1])[::-1]
    since = np.clip(idx - last, 0, 3600)[centers]
    until = np.clip(nxt - idx, 0, 3600)[centers]

    # --- activity counts на секундной сетке
    cnt = np.interp(t0 + idx, counts[:, 0], counts[:, 1])
    # --- ЧСС на секундной сетке, нормировка на человека (медиана/IQR за ночь)
    hr1 = np.interp(t0 + idx, hr[:, 0], hr[:, 1])
    night = (t0 + idx >= te.min()) & (t0 + idx <= te.max() + 30)
    med = np.median(hr1[night])
    iqr = np.subtract(*np.percentile(hr1[night], [75, 25])) or 1.0
    hrz = (hr1 - med) / iqr

    cs = lambda x: np.concatenate([[0.0], np.cumsum(x)])
    c_cnt, c_enmo, c_dang, c_act, c_hr, c_hr2 = map(cs, [cnt, enmo, dangle, active, hrz, hrz ** 2])

    feats, names = [], []

    def add(name, values):
        feats.append(np.asarray(values, float))
        names.append(name)

    add("paper_motion", paper["motion"])
    add("paper_hr", paper["hr"])
    add("paper_clock", paper["clock"])
    for w in WINDOWS_MIN:
        h = int(w * 60)
        add(f"counts_{w}m", np.log1p(_window_mean(c_cnt, centers, h)))
        add(f"enmo_{w}m", np.log1p(1000 * _window_mean(c_enmo, centers, h)))
        add(f"dangle_{w}m", np.log1p(_window_mean(c_dang, centers, h)))
        add(f"active_frac_{w}m", _window_mean(c_act, centers, h))
        m = _window_mean(c_hr, centers, h)
        add(f"hr_level_{w}m", m)
        add(f"hr_std_{w}m", np.sqrt(np.clip(_window_mean(c_hr2, centers, h) - m ** 2, 0, None)))
    for a, b, tag in [(-1200, 0, "past20"), (0, 1200, "next20"), (-600, 0, "past10"), (30, 630, "next10")]:
        add(f"counts_{tag}", np.log1p(_window_range(c_cnt, centers, a, b)))
        add(f"active_frac_{tag}", _window_range(c_act, centers, a, b))
        add(f"hr_level_{tag}", _window_range(c_hr, centers, a, b))
    add("hr_trend_10m", _window_range(c_hr, centers, 0, 600) - _window_range(c_hr, centers, -600, 0))
    add("log_since_move", np.log1p(since))
    add("log_until_move", np.log1p(until))
    add("time_h", (te - te.min()) / 3600)

    X = np.column_stack(feats)
    np.savez(path, X=X, names=np.array(names))
    return dict(X=X, names=names)


def load_all(paper_data):
    out = {}
    for s in sl.SUBJECTS:
        out[s] = build_subject(s, paper_data[s])
    names = out[sl.SUBJECTS[0]]["names"]
    return {s: out[s]["X"] for s in sl.SUBJECTS}, names
