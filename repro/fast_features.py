"""Три «быстрых» признака команды (документ «MLP: исходные признаки и три быстрых показателя»).

Для каждой эпохи 30 с [t, t + 30):
  m_k      — мера движения на 5-секундном отрезке k: m = sqrt(v_x + v_y + v_z), v_q — дисперсия ускорения по оси q
             (в g); отрезок доступен, если в нём не меньше двух измерений
  f_peak   — пик движения за эпоху: ln(1 + max_{k ∈ C} m_k / τ), C — доступные отрезки эпохи (6 отрезков)
  f_active — доля активных отрезков в минутном окне W_t = [t − 15, t + 45) (12 отрезков):
             Σ A_k · 1{m_k ≥ τ} / Σ A_k, A_k — доступность отрезка
  f_hr     — относительный пульс за минуту: (среднее пульса в W_t − медиана пульса записи) / медиана
τ = 0.02 g. Признаки считаются по обрезанным сырым данным (outputs/cropped/) для тех же валидных эпох, что и
признаки статьи (RawDataProcessor.get_valid_epochs), поэтому строки совпадают с outputs/features/ один в один.
Окно W_t заглядывает на 15 с в будущее, а медиана пульса берётся по всей записи — это анализ готовой записи,
а не онлайн-предсказание. Считает эталонная функция команды make_fast_features из
additional_sleep_features/additional_sleep_features.py (окна обрезаются по непрерывному блоку эпох); если её нет
или в окне нет данных — своя реализация ниже (нет данных: f_peak = 0, f_active = 0, f_hr = 0).

Кэш: outputs/repro/fast_features/<id>.npz. Запуск отдельно (считает и кэширует для всех):
    python repro/fast_features.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CROPPED = ROOT / "outputs" / "cropped"
CACHE = ROOT / "outputs" / "repro" / "fast_features"
FAST_FEATURES = ["peak", "active", "hr_rel"]
TAU = 0.02          # g
SEGMENT = 5.0       # с
EPOCH = 30.0        # с
WINDOW = (-15.0, 45.0)  # минутное окно относительно начала эпохи, с


def segment_motion(acc, origin):
    """m_k и доступность для всех 5-секундных отрезков, отсчитанных от origin (начало PSG)."""
    idx = np.floor((acc[:, 0] - origin) / SEGMENT).astype(np.int64)
    keep = idx >= 0
    idx, xyz = idx[keep], acc[keep, 1:4]
    n_seg = int(idx.max()) + 1 if len(idx) else 0
    count = np.bincount(idx, minlength=n_seg).astype(float)
    var_sum = np.zeros(n_seg)
    for q in range(3):
        s1 = np.bincount(idx, weights=xyz[:, q], minlength=n_seg)
        s2 = np.bincount(idx, weights=xyz[:, q] ** 2, minlength=n_seg)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / count
            var_sum += np.maximum(s2 / count - mean ** 2, 0.0)  # дисперсия с делением на N, как в документе
    available = count >= 2
    m = np.where(available, np.sqrt(var_sum), 0.0)
    return m, available


def team_make_fast_features():
    """Эталонная функция команды: additional_sleep_features/additional_sleep_features.py (Ян Мусатов)."""
    import importlib.util
    path = ROOT / "additional_sleep_features" / "additional_sleep_features.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("additional_sleep_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_fast_features


def compute(subject):
    """Признаки по эталонной функции команды; если её нет — по собственной реализации ниже."""
    from source.preprocessing.raw_data_processor import RawDataProcessor

    epochs = [e.timestamp for e in RawDataProcessor.get_valid_epochs(subject)]
    make = team_make_fast_features()
    if make is not None:
        acc = np.loadtxt(CROPPED / f"{subject}_cleaned_motion.out")
        hr = np.loadtxt(CROPPED / f"{subject}_cleaned_hr.out")
        acc, hr = acc[np.argsort(acc[:, 0], kind="stable")], hr[np.argsort(hr[:, 0], kind="stable")]
        try:
            feats, _ = make(acc, hr, np.array(epochs, dtype=float), TAU)
            return {"peak": feats[:, 0], "active": feats[:, 1], "hr_rel": feats[:, 2],
                    "epoch_timestamp": np.array(epochs)}
        except ValueError as error:  # нет данных в каком-то окне: считаем своей реализацией (там 0)
            print(f"{subject}: эталонная функция команды не справилась ({error}) — своя реализация")
    return compute_own(subject, epochs)


def compute_own(subject, epochs):
    psg = np.loadtxt(CROPPED / f"{subject}_cleaned_psg.out")
    origin = psg[0, 0]
    acc = np.loadtxt(CROPPED / f"{subject}_cleaned_motion.out")
    hr = np.loadtxt(CROPPED / f"{subject}_cleaned_hr.out")
    m, available = segment_motion(acc, origin)
    n_seg = len(m)
    hr_median = np.median(hr[:, 1])
    hr_t = hr[:, 0]
    order = np.argsort(hr_t)
    hr_t, hr_v = hr_t[order], hr[order, 1]
    hr_cum = np.concatenate([[0.0], np.cumsum(hr_v)])

    def segs(start, count):
        k = np.arange(count) + int(round((start - origin) / SEGMENT))
        ok = (k >= 0) & (k < n_seg)
        k = k[ok]
        return m[k], available[k]

    peak, active, hr_rel = [], [], []
    for t in epochs:
        mk, ak = segs(t, int(EPOCH / SEGMENT))
        peak.append(np.log1p(mk[ak].max() / TAU) if ak.any() else 0.0)
        mk, ak = segs(t + WINDOW[0], int((WINDOW[1] - WINDOW[0]) / SEGMENT))
        active.append((ak & (mk >= TAU)).sum() / ak.sum() if ak.any() else 0.0)
        lo, hi = np.searchsorted(hr_t, t + WINDOW[0], "left"), np.searchsorted(hr_t, t + WINDOW[1], "left")
        hr_rel.append(((hr_cum[hi] - hr_cum[lo]) / (hi - lo) - hr_median) / hr_median if hi > lo else 0.0)
    return {"peak": np.array(peak), "active": np.array(active), "hr_rel": np.array(hr_rel),
            "epoch_timestamp": np.array(epochs)}


def load(subject, refresh=False):
    """Быстрые признаки одного человека (из кэша или посчитать)."""
    path = CACHE / f"{subject}.npz"
    if path.exists() and not refresh:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    if not (CROPPED / f"{subject}_cleaned_motion.out").exists():
        if not (ROOT / "data" / "motion" / f"{subject}_acceleration.txt").exists():
            raise SystemExit("нет сырых данных в data/ — скачайте датасет: python repro/download_data.py")
        from source.preprocessing.raw_data_processor import RawDataProcessor
        CROPPED.mkdir(parents=True, exist_ok=True)
        RawDataProcessor.crop_all(subject)  # обрезка сырых данных по общему интервалу (признаки статьи не меняет)
    feats = compute(subject)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **feats)
    return feats


def add_to(data, refresh=False):
    """Добавляет peak, active, hr_rel в словарь sleeplib.load_paper_features() (проверяя число эпох)."""
    for subject, d in data.items():
        feats = load(subject, refresh)
        n = len(d["stage"])
        if len(feats["peak"]) != n:
            raise SystemExit(f"{subject}: быстрых признаков {len(feats['peak'])}, а эпох в outputs/features {n}. "
                             f"Пересчитайте: python repro/fast_features.py --refresh")
        for name in FAST_FEATURES:
            d[name] = feats[name]
    return data


def main():
    sys.path.insert(0, str(ROOT / "repro"))
    import sleeplib as sl
    from progress import Progress

    refresh = "--refresh" in sys.argv
    subjects = [a for a in sys.argv[1:] if not a.startswith("--")] or sl.SUBJECTS
    bar = Progress(len(subjects), "быстрые признаки")
    for s in subjects:
        f = load(s, refresh)
        bar.update(status=f"{s}: {len(f['peak'])} эпох, peak {np.mean(f['peak']):.2f}, "
                          f"active {np.mean(f['active']):.2f}, hr_rel {np.mean(f['hr_rel']):+.3f}")
    bar.close()
    print("кэш:", CACHE.relative_to(ROOT))


if __name__ == "__main__":
    main()
