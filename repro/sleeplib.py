"""Общие функции для воспроизведения и расширенного анализа (repro/*.py).

Читает признаки, которые строит оригинальный препроцессинг (outputs/features/*.out), и реализует
те же метрики, что и статья, но быстро, векторно и с фиксированным seed.

Обозначения меток: sleep/wake — 0 = wake, 1 = sleep;  три класса — 0 = wake, 1 = NREM, 2 = REM.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "outputs" / "features"
CROPPED_DIR = ROOT / "outputs" / "cropped"
OUT = ROOT / "outputs" / "repro"

SUBJECTS = ['3509524', '5132496', '1066528', '5498603', '2638030', '2598705', '5383425', '1455390', '4018081',
            '9961348', '1449548', '8258170', '781756', '9106476', '8686948', '8530312', '3997827', '4314139',
            '1818471', '4426783', '8173033', '7749105', '5797046', '759667', '8000685', '6220552', '844359',
            '9618981', '1360686', '46343', '8692923']

# Имя признака в отчёте -> суффикс файла outputs/features/<id>_<suffix>_feature.out
PAPER_FEATURES = {"motion": "count", "hr": "hr", "clock": "circadian", "cosine": "cosine", "time": "time"}

PAPER_FEATURE_SETS = {
    "Motion only": ["motion"],
    "HR only": ["hr"],
    "Motion, HR": ["motion", "hr"],
    "Motion, HR, Clock": ["motion", "hr", "clock"],
}
FEATURE_SET_COLORS = {"Motion only": "#3b5b92", "HR only": "#fcb001", "Motion, HR": "#39ad48",
                      "Motion, HR, Clock": "#d3494e", "Motion, HR, Cosine": "#580f41", "Motion, HR, Time": "#a8a495"}


def sleep_wake(stages):
    return (np.asarray(stages) > 0).astype(int)


def three_class(stages):
    stages = np.asarray(stages)
    return np.where(stages == 0, 0, np.where(stages == 5, 2, 1))


def load_paper_features(subjects=SUBJECTS):
    """{subject: {"stage": (n,), "motion": (n,), "hr": (n,), "clock": (n,), ...}}

    Все строки читаются целиком (оригинальный код через pd.read_csv() теряет первую эпоху как «заголовок»
    и у признаков, и у меток — согласованно, так что на результат это почти не влияет).
    """
    data = {}
    for s in subjects:
        d = {"stage": np.loadtxt(FEATURES_DIR / f"{s}_psg_labels.out")}
        for name, suffix in PAPER_FEATURES.items():
            path = FEATURES_DIR / f"{s}_{suffix}_feature.out"
            if path.exists():
                d[name] = np.loadtxt(path).reshape(-1)
        n = len(d["stage"])
        assert all(len(v) == n for v in d.values()), f"length mismatch for {s}"
        data[s] = d
    return data


def stack(data, subjects, features):
    X = np.vstack([np.column_stack([data[s][f] for f in features]) for s in subjects])
    y = np.concatenate([data[s]["stage"] for s in subjects])
    groups = np.concatenate([[s] * len(data[s]["stage"]) for s in subjects])
    return X, y, groups


def monte_carlo_splits(subjects, n_splits, test_fraction=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n_test = int(round(test_fraction * len(subjects)))
    for _ in range(n_splits):
        perm = rng.permutation(subjects)
        yield list(perm[n_test:]), list(perm[:n_test])


def loso_splits(subjects):
    for s in subjects:
        yield [t for t in subjects if t != s], [s]


# ---------------------------------------------------------------- sleep / wake metrics (Tables 2–5)

def sw_metrics_at_sensitivity(y_sleep, p_sleep, sensitivities=(0.8, 0.9, 0.93, 0.95)):
    """Порог выбирается так, чтобы доля верно распознанного сна = заданной (как в Таблицах 2–5)."""
    y_sleep = np.asarray(y_sleep)
    rows = []
    sleep_scores = np.sort(p_sleep[y_sleep == 1])
    for target in sensitivities:
        threshold = np.quantile(sleep_scores, 1 - target)  # доля сна с p >= threshold == target
        pred = (p_sleep >= threshold).astype(int)
        rows.append(dict(
            sensitivity=target,
            accuracy=np.mean(pred == y_sleep),
            wake_correct=np.mean(pred[y_sleep == 0] == 0),
            sleep_correct=np.mean(pred[y_sleep == 1] == 1),
            kappa=cohen_kappa_score(y_sleep, pred),
            threshold=threshold,
        ))
    return rows


def sw_summary(y_sleep, p_sleep):
    rows = sw_metrics_at_sensitivity(y_sleep, p_sleep)
    out = {"auc": roc_auc_score(y_sleep, p_sleep)}
    for r in rows:
        tag = f"@{int(round(r['sensitivity'] * 100))}"
        out[f"wake_correct{tag}"] = r["wake_correct"]
        out[f"accuracy{tag}"] = r["accuracy"]
        out[f"kappa{tag}"] = r["kappa"]
    return out


def interp_roc(y_sleep, p_sleep, grid=np.linspace(0, 1, 101)):
    """ROC «доля wake, ошибочно названная sleep» -> «доля sleep, названная sleep», на общей сетке."""
    fpr, tpr, _ = roc_curve(y_sleep, p_sleep, drop_intermediate=False)
    return np.interp(grid, fpr, tpr)


# ---------------------------------------------------------------- three-class metrics (Table 6, Figure 4)

def three_class_at_wake_fpr(y3, proba, wake_fpr):
    """Двухпороговая схема статьи.

    1) порог на P(wake) подбирается так, чтобы доля wake, названная сном, = wake_fpr;
    2) среди эпох, названных сном, порог на P(REM)/(P(REM)+P(NREM)) подбирается так, чтобы
       доли верно распознанных NREM и REM были как можно ближе.
    Возвращает dict с точностями классов, общей accuracy и kappa, а также лучшую accuracy по сетке
    REM-порогов (как «Best accuracy» в Таблице 6).
    """
    y3 = np.asarray(y3)
    p_wake = proba[:, 0]
    wake_scores = p_wake[y3 == 0]
    # эпоха — сон, если p_wake < thr; доля wake со p_wake < thr == wake_fpr
    thr_wake = np.quantile(wake_scores, wake_fpr) if 0 < wake_fpr < 1 else (-np.inf if wake_fpr == 0 else np.inf)
    is_sleep = p_wake < thr_wake if wake_fpr < 1 else np.ones_like(p_wake, bool)
    rem_ratio = proba[:, 2] / np.clip(proba[:, 1] + proba[:, 2], 1e-12, None)

    def evaluate(thr_rem):
        pred = np.where(~is_sleep, 0, np.where(rem_ratio >= thr_rem, 2, 1))
        acc_c = [np.mean(pred[y3 == c] == c) if np.any(y3 == c) else np.nan for c in range(3)]
        return pred, acc_c

    grid = np.unique(np.quantile(rem_ratio, np.linspace(0, 1, 201)))
    best = None
    best_acc = (-1, None, None)
    for thr in grid:
        pred, acc_c = evaluate(thr)
        gap = abs(acc_c[1] - acc_c[2])
        if best is None or gap < best[0]:
            best = (gap, thr, pred, acc_c)
        acc = np.mean(pred == y3)
        if acc > best_acc[0]:
            best_acc = (acc, thr, pred)
    _, thr, pred, acc_c = best
    return dict(wake_fpr=wake_fpr, wake_correct=acc_c[0], nrem_correct=acc_c[1], rem_correct=acc_c[2],
                balanced_accuracy=np.mean(pred == y3), balanced_kappa=cohen_kappa_score(y3, pred),
                best_accuracy=best_acc[0], best_kappa=cohen_kappa_score(y3, best_acc[2]))


def fixed_threshold_predict(proba, theta_wake=0.3, theta_rem=0.35):
    """Фиксированные пороги, как в Bland–Altman (рис. 5): wake если P(wake) > θW, иначе REM если P(REM) > θREM."""
    return np.where(proba[:, 0] > theta_wake, 0, np.where(proba[:, 2] > theta_rem, 2, 1))


def sleep_metrics(y3, epoch_minutes=0.5):
    """TST, SOL, WASO, SE, REM и NREM (минуты) для одной ночи по трёхклассовой последовательности."""
    y3 = np.asarray(y3)
    asleep = np.where(y3 > 0)[0]
    tib = len(y3) * epoch_minutes
    tst = len(asleep) * epoch_minutes
    sol = (asleep[0] if len(asleep) else len(y3)) * epoch_minutes
    waso = (np.sum(y3[asleep[0]:] == 0) * epoch_minutes) if len(asleep) else 0.0
    return dict(TST=tst, SOL=sol, WASO=waso, SE=tst / tib, REM=np.sum(y3 == 2) * epoch_minutes,
                NREM=np.sum(y3 == 1) * epoch_minutes)
