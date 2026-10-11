"""ВСЕ метрики статьи Walch et al. (2019) + F1, AP и PPV — для четырёх моделей статьи и CatBoost.

    python repro/09_paper_metrics.py                          # все 5 моделей (дольше всех — нейросеть и CatBoost)
    python repro/09_paper_metrics.py --models "Neural Net"    # обучить только нейросеть; остальные — из сохранённых
    python repro/09_paper_metrics.py --skip-nn                # все, кроме нейросети
    python repro/09_paper_metrics.py --plots-only             # ничего не обучать, перерисовать из сохранённых
    (из корня репозитория; в VS Code — конфигурация «9. All paper metrics»; всё сразу — python repro/run_all.py)

Обучаются модели из --models; в таблицы и на рисунки попадают ВСЕ модели, чьи предсказания уже сохранены
(outputs/repro/paper_metrics/oof/). Во время обучения показываются этап (модель, признаки, задача) и прогресс-бар
по фолдам с оценкой оставшегося времени. В конце всё собирается в outputs/RESULTS.md (08_results_markdown.py).

Протокол: leave-one-subject-out (31 фолд) — каждый человек предсказывается моделью, обученной на остальных 30.
Модели — из кода статьи (source/utils.py), фиксированные значения из её сеток перебора:
  Logistic Regression  liblinear, L2, C = 0.001 (сон/бодрствование) / 0.1 (три класса), веса классов balanced,
                       три класса — one-vs-rest, как в sklearn 0.20
  k-Nearest Neighbors  k = 500, weights = distance
  Random Forest        100 деревьев, max_depth 10, min_samples_leaf 32, max_features 1.0, веса классов balanced
  Neural Net           MLP (15, 15, 15), relu, adam, alpha 0.01, max_iter 2000, n_iter_no_change 20
  CatBoost             параметры из source/analysis/classification/catboost_config.py (FAST / полный режим),
                       веса классов balanced; нужен пакет catboost (run_all.py ставит его сам)
Признаки — как в статье: Motion only / HR only / Motion, HR / Motion, HR, Clock.

Что получается (outputs/repro/paper_metrics/):
  table1_*                 Таблица 1: TST, TIB, SOL, WASO, SE, REM, NREM по PSG, рядом — статья
  sleep_wake_all_models    Таблицы 2–5: при «сон найден» 0.80/0.90/0.93/0.95 — accuracy, wake найден, сон найден, κ,
                           AUC + AP wake, PPV wake/сон, F1 wake/сон, macro F1
  three_class_all_models   Таблица 6 + macro F1 и F1 по стадиям
  bland_altman_*           Рис. 5: сдвиг, SD и границы согласия TST, SOL, WASO, SE, REM, NREM — для каждой модели
  per_subject_*            Рис. 6–7: качество по каждому человеку — для каждой модели
  fig2–fig4, fig_summary   аналоги рисунков 2–4 и сводка по всем моделям
  fig5–fig7__<модель>      рисунки 5–7 для каждой модели (fig5–fig7 без суффикса — по нейросети, как в статье)
  fig_new_metrics, fig_f1_threshold_models   новые метрики: AP, PPV, F1, F1 в зависимости от порога
Важность признаков по Шепли — для всех моделей (сон/бодрствование):
  shapley_subsets, shapley_values   точное разложение Шепли качества (AUC, AP wake, κ, F1 wake) по признакам
                           motion, hr, clock: модели переобучаются на всех 7 подмножествах признаков (LOSO);
                           рядом — drop-column (сколько теряем без признака) и качество признака в одиночку
  shap_mean_abs            SHAP-значения по эпохам: на сколько каждый признак сдвигает P(сон) у отложенного
                           человека (точный перебор 2^3 коалиций, одинаковый для всех моделей)
  fig_shapley_auc, fig_shapley_metrics, fig_shapley_share, fig_shap_beeswarm_models, fig_shap_mean_abs,
  fig_shap_dependence_models
Таблицы 7–9 и рис. 8–9 (MESA) не воспроизводимы: данные MESA выдаются только по заявке NSRR.
"""
import argparse
import itertools
import math
import os
import subprocess
import sys
import time
import warnings
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, cohen_kappa_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(REPRO.parent))
import sleeplib as sl  # noqa: E402
from progress import Progress, fmt_time  # noqa: E402
from source.analysis.classification import catboost_config  # noqa: E402
from source.analysis.classification.catboost_classifier import CatBoostSleepClassifier, catboost_available  # noqa: E402
from source.utils import LiblinearLogisticRegression  # noqa: E402

warnings.filterwarnings("ignore")
N_JOBS = int(os.environ.get("SLEEP_N_JOBS", 6))
SENSITIVITIES = (0.80, 0.90, 0.93, 0.95)
THETA_WAKE, THETA_REM = 0.3, 0.35  # фиксированные пороги статьи (рис. 5, 6)
SMOKE = False  # --smoke: 6 человек и крошечные модели — только проверить, что всё запускается

CLASSIFIERS = ["Logistic Regression", "k-Nearest Neighbors", "Random Forest", "Neural Net", "CatBoost"]
NAMES_RU = {"Logistic Regression": "логистическая регрессия", "k-Nearest Neighbors": "k ближайших соседей",
            "Random Forest": "случайный лес", "Neural Net": "нейросеть", "CatBoost": "CatBoost"}
FEATURE_SETS = {"Motion only": ["motion"], "HR only": ["hr"], "Motion, HR": ["motion", "hr"],
                "Motion, HR, Clock": ["motion", "hr", "clock"]}
SHAPLEY_FEATURES = ["motion", "hr", "clock"]
# ещё 3 подмножества — вместе с наборами статьи это все 7 подмножеств {motion, hr, clock}, нужные для точного
# разложения Шепли (только сон/бодрствование; в таблицы и рисунки статьи не идут)
EXTRA_SETS = {"Clock only": ["clock"], "Motion, Clock": ["motion", "clock"], "HR, Clock": ["hr", "clock"]}
ALL_SETS = {**FEATURE_SETS, **EXTRA_SETS}
SHAPLEY_METRICS = {"auc": "AUC", "ap_wake": "AP wake", "kappa": "κ при 90% сна", "f1_wake": "F1 wake при 90% сна"}
SHAP_STAGE = "SHAP"  # этап обучения: модель на Motion, HR, Clock + SHAP-значения отложенного человека
TARGETS = {"sw": "сон/бодрствование", "3": "3 класса"}
COLORS = {"Motion only": "#3b5b92", "HR only": "#f0a202", "Motion, HR": "#39ad48", "Motion, HR, Clock": "#d3494e"}
MODEL_COLORS = {"Logistic Regression": "#4c72b0", "k-Nearest Neighbors": "#8c8c8c", "Random Forest": "#55a868",
                "Neural Net": "#c44e52", "CatBoost": "#dd8452"}
PAPER_TABLE1 = {  # Таблица 1 статьи: среднее (SD)
    "TST, мин": (427.87, 38.87), "TIB, мин": (472.56, 27.03), "SOL, мин": (14.97, 10.1), "WASO, мин": (28.73, 22.8),
    "SE, %": (90.48, 5.54), "REM, мин": (107.15, 31.22), "NREM, мин": (320.77, 39.11)}


def paths():
    out = sl.OUT / ("smoke" if SMOKE else "") / "paper_metrics"
    return out, out / "oof"


def make_model(name, target, smoke=False):
    """smoke передаётся явно: модель строится в процессе-работнике joblib, где глобальные флаги не видны."""
    if name == "Logistic Regression":
        return LiblinearLogisticRegression(penalty="l2", solver="liblinear", C=0.001 if target == "sw" else 0.1,
                                           class_weight="balanced")
    if name == "k-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=20 if smoke else 500, weights="distance")
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=10 if smoke else 100, max_features=1.0, max_depth=10,
                                      min_samples_split=10, min_samples_leaf=32, class_weight="balanced", n_jobs=1,
                                      random_state=0)
    if name == "Neural Net":
        return MLPClassifier(activation="relu", hidden_layer_sizes=(15, 15, 15), alpha=0.01, solver="adam",
                             max_iter=30 if smoke else 2000, n_iter_no_change=20, random_state=0)
    params = dict(catboost_config.PARAMS)
    if smoke:
        params.update(iterations=10, early_stopping_rounds=None, refit_on_full=False)
    return CatBoostSleepClassifier(class_weight="balanced", **params)


def slug(text):
    return text.lower().replace(", ", "_").replace(" ", "_").replace("-", "_")


def cache_path(clf, fs, target):
    return paths()[1] / f"{slug(clf)}__{slug(fs)}__{target}.npz"


def cached(clf):
    return all(cache_path(clf, fs, t).exists() for fs in FEATURE_SETS for t in TARGETS)


# --------------------------------------------------------------------------- обучение (LOSO) с кэшем и прогрессом
def _fold(data, clf, fs, target, test, subjects, smoke):
    """Один фолд LOSO. Список людей и режим передаются явно: процесс-работник joblib импортирует модули заново."""
    conv = sl.sleep_wake if target == "sw" else sl.three_class
    features = ALL_SETS[fs]
    X, y, _ = sl.stack(data, [s for s in subjects if s != test], features)
    model = make_model(clf, target, smoke).fit(X, conv(y))
    return test, model.predict_proba(sl.stack(data, [test], features)[0])


def loso_predictions(data, clf, fs, target, bar=None, status=""):
    """Out-of-fold вероятности всех людей; результат кэшируется в .npz."""
    path = cache_path(clf, fs, target)
    if path.exists():
        z = np.load(path)
        return {s: z[s] for s in sl.SUBJECTS}
    subjects = list(sl.SUBJECTS)
    result = {}
    for subject, proba in parallel_stream(delayed(_fold)(data, clf, fs, target, s, subjects, SMOKE)
                                          for s in subjects):
        result[subject] = proba
        if bar:
            bar.update(status=f"{status} | фолд {len(result)}/{len(sl.SUBJECTS)}")
    np.savez(path, **result)
    return result


def parallel_stream(jobs):
    try:  # joblib >= 1.4: результаты по мере готовности -> прогресс по фолдам
        return Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(jobs)
    except TypeError:  # старый joblib: все фолды разом
        return Parallel(n_jobs=N_JOBS)(jobs)


# --------------------------------------------------------------------------- важность признаков: Шепли
def shapley_cached(clf):
    return all(cache_path(clf, fs, "sw").exists() for fs in ALL_SETS)


def shap_path(clf):
    return paths()[1] / f"{slug(clf)}__shap.npz"


def exact_shap(model, X, background):
    """Точные интервенционные SHAP-значения P(сон): перебор всех 2^d коалиций признаков; признаки вне коалиции
    берутся из фоновой выборки (обучающие эпохи) и усредняются. Одинаково для любой модели, единицы — P(сон)."""
    n, d = X.shape
    value = {}
    for mask in itertools.product([False, True], repeat=d):
        keep = np.array(mask)
        Z = np.repeat(background[None], n, axis=0)  # (n, фон, d)
        Z[:, :, keep] = X[:, None, keep]
        value[mask] = model.predict_proba(Z.reshape(-1, d))[:, 1].reshape(n, len(background)).mean(1)
    phi = np.zeros((n, d))
    for j in range(d):
        for mask, v in value.items():
            if not mask[j]:
                size = sum(mask)
                weight = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
                phi[:, j] += weight * (value[mask[:j] + (True,) + mask[j + 1:]] - v)
    return phi


def _shap_fold(data, clf, test, subjects, smoke):
    """Модель на Motion, HR, Clock без человека test; SHAP для случайных эпох test (out-of-fold)."""
    X, y, _ = sl.stack(data, [s for s in subjects if s != test], SHAPLEY_FEATURES)
    model = make_model(clf, "sw", smoke).fit(X, sl.sleep_wake(y))
    rng = np.random.default_rng(zlib.crc32(str(test).encode()))
    Xt = sl.stack(data, [test], SHAPLEY_FEATURES)[0]
    n_explain, n_background = (30, 10) if smoke else (150, 50)
    Xt = Xt[np.sort(rng.choice(len(Xt), min(n_explain, len(Xt)), replace=False))]
    background = X[rng.choice(len(X), min(n_background, len(X)), replace=False)]
    return test, exact_shap(model, Xt, background), Xt


def shap_loso(data, clf, bar=None, status=""):
    """SHAP-значения по всем людям (LOSO); кэшируются в .npz. Возвращает (значения, признаки) — массивы (n, 3)."""
    path = shap_path(clf)
    if path.exists():
        z = np.load(path)
        return z["values"], z["X"]
    subjects = list(sl.SUBJECTS)
    result = {}
    for subject, values, X in parallel_stream(delayed(_shap_fold)(data, clf, s, subjects, SMOKE) for s in subjects):
        result[subject] = (values, X)
        if bar:
            bar.update(status=f"{status} | фолд {len(result)}/{len(sl.SUBJECTS)}")
    values = np.vstack([result[s][0] for s in subjects])
    X = np.vstack([result[s][1] for s in subjects])
    np.savez(path, values=values, X=X, subject=np.repeat(subjects, [len(result[s][0]) for s in subjects]))
    return values, X


def shapley_baseline(metric, y):
    """Качество модели без признаков (случайной): от него отсчитывается вклад признаков."""
    wake = np.mean(y == 0)
    # F1 wake при 90% сна: случайная модель говорит «wake» в 10% эпох -> recall 0.1, precision = доля wake
    return {"auc": 0.5, "ap_wake": wake, "kappa": 0.0, "f1_wake": 2 * wake * 0.1 / (wake + 0.1)}[metric]


def shapley_tables(data, models, y_all):
    """Точное разложение Шепли: игра v(S) = метрика модели на признаках S − метрика случайной модели.
    Значение Шепли признака — его средний прирост v по всем порядкам добавления признаков; сумма по признакам
    равна v(все признаки). Возвращает качество на 7 подмножествах и значения Шепли."""
    subset_rows, value_rows = [], []
    n = len(SHAPLEY_FEATURES)
    for clf in models:
        quality = {}
        for fs, features in ALL_SETS.items():
            p = loso_predictions(data, clf, fs, "sw")
            m = next(r for r in sleep_wake_metrics(y_all, np.concatenate([p[s][:, 1] for s in sl.SUBJECTS]))
                     if r["sensitivity_target"] == 0.9)
            quality[frozenset(features)] = m
            subset_rows.append(dict(classifier=clf, feature_set=fs, n_features=len(features),
                                    **{k: m[k] for k in SHAPLEY_METRICS}))
        for metric in SHAPLEY_METRICS:
            base = shapley_baseline(metric, y_all)
            v = {S: m[metric] - base for S, m in quality.items()}
            v[frozenset()] = 0.0
            full = v[frozenset(SHAPLEY_FEATURES)]
            for f in SHAPLEY_FEATURES:
                others = [g for g in SHAPLEY_FEATURES if g != f]
                phi = sum(math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                          * (v[frozenset(S) | {f}] - v[frozenset(S)])
                          for r in range(n) for S in itertools.combinations(others, r))
                value_rows.append(dict(classifier=clf, metric=metric, feature=f, shapley=phi,
                                       share=phi / full if full else np.nan, drop_column=full - v[frozenset(others)],
                                       alone=v[frozenset([f])], full_model=full + base, baseline=base))
    return pd.DataFrame(subset_rows), pd.DataFrame(value_rows)


def shap_summary(shap_values):
    """mean |SHAP|, доля и направление (Спирмен между значением признака и его SHAP) для каждой модели."""
    rows = []
    for clf, (values, X) in shap_values.items():
        total = np.abs(values).mean(0).sum()
        for j, f in enumerate(SHAPLEY_FEATURES):
            rho = pd.Series(X[:, j]).corr(pd.Series(values[:, j]), method="spearman")
            rows.append(dict(classifier=clf, feature=f, mean_abs=np.abs(values[:, j]).mean(),
                             share=np.abs(values[:, j]).mean() / total if total else np.nan,
                             mean=values[:, j].mean(), spearman_value_vs_shap=rho, n_epochs=len(values)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- метрики
def sleep_wake_metrics(y, p_sleep):
    """Таблицы 2–5 + AP, PPV, F1 на всех эпохах (out-of-fold)."""
    rows = []
    auc = roc_auc_score(y, p_sleep)
    ap = average_precision_score(1 - y, 1 - p_sleep)  # wake как положительный класс (рис. 3)
    for target in SENSITIVITIES:
        threshold = np.quantile(p_sleep[y == 1], 1 - target)
        pred = (p_sleep >= threshold).astype(int)
        rows.append(dict(sensitivity_target=target, accuracy=np.mean(pred == y),
                         wake_correct=np.mean(pred[y == 0] == 0), sleep_correct=np.mean(pred[y == 1] == 1),
                         kappa=cohen_kappa_score(y, pred), auc=auc, ap_wake=ap,
                         ppv_wake=np.mean(y[pred == 0] == 0) if np.any(pred == 0) else np.nan,
                         ppv_sleep=np.mean(y[pred == 1] == 1), **sl.sw_f1(y, pred)))
    return rows


def table1(data):
    """Аналог Таблицы 1 статьи по разметке PSG (все размеченные эпохи, если есть data/labels)."""
    rows = []
    for s in sl.SUBJECTS:
        path = sl.ROOT / "data" / "labels" / f"{s}_labeled_sleep.txt"
        stages = np.loadtxt(path)[:, 1] if path.exists() else data[s]["stage"]
        stages = stages[stages >= 0]
        m = sl.sleep_metrics(sl.three_class(stages))
        rows.append({"subject": s, "TST, мин": m["TST"], "TIB, мин": len(stages) * 0.5, "SOL, мин": m["SOL"],
                     "WASO, мин": m["WASO"], "SE, %": 100 * m["SE"], "REM, мин": m["REM"], "NREM, мин": m["NREM"]})
    per = pd.DataFrame(rows)
    summary = [{"показатель": col, "наши: среднее": per[col].mean(), "наши: SD": per[col].std(),
                "наши: мин": per[col].min(), "наши: макс": per[col].max(), "статья: среднее": pm, "статья: SD": psd}
               for col, (pm, psd) in PAPER_TABLE1.items()]
    source = "data/labels" if (sl.ROOT / "data" / "labels").exists() else "outputs/features (валидные эпохи)"
    return per, pd.DataFrame(summary), source


def bland_altman(probas3, y3s):
    """Рис. 5: показатели сна PSG и модели для каждого человека при фиксированных порогах статьи."""
    rows = []
    for s in sl.SUBJECTS:
        pred = sl.fixed_threshold_predict(probas3[s], THETA_WAKE, THETA_REM)
        true_m, pred_m = sl.sleep_metrics(y3s[s]), sl.sleep_metrics(pred)
        for key in ["TST", "SOL", "WASO", "SE", "REM", "NREM"]:
            rows.append(dict(subject=s, metric=key, psg=true_m[key], model=pred_m[key], diff=true_m[key] - pred_m[key]))
    return pd.DataFrame(rows)


def per_subject(p_sw, y_sw):
    """Рис. 6 (порог P(wake) ≥ 0.3) и рис. 7 (порог под каждого человека)."""
    per6, per7 = [], []
    for s in sl.SUBJECTS:
        y = y_sw[s]
        pred = np.where(p_sw[s][:, 0] >= THETA_WAKE, 0, 1)
        has_wake = np.any(y == 0)
        per6.append(dict(subject=s, accuracy=np.mean(pred == y),
                         wake_correct=np.mean(pred[y == 0] == 0) if has_wake else np.nan,
                         sleep_correct=np.mean(pred[y == 1] == 1), kappa=cohen_kappa_score(y, pred),
                         f1_wake=sl.sw_f1(y, pred)["f1_wake"] if has_wake else np.nan))
        for tpr in SENSITIVITIES:
            threshold = np.quantile(p_sw[s][y == 1, 1], 1 - tpr)
            pred = (p_sw[s][:, 1] >= threshold).astype(int)
            per7.append(dict(subject=s, tpr=tpr, accuracy=np.mean(pred == y),
                             wake_correct=np.mean(pred[y == 0] == 0) if has_wake else np.nan))
    return pd.DataFrame(per6), pd.DataFrame(per7)


# --------------------------------------------------------------------------- рисунки
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25)


def model_grid(models):
    cols = 3 if len(models) > 4 else 2
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")  # моделей меньше, чем ячеек
    return fig, axes


def save(fig, name):
    fig.tight_layout()
    fig.savefig(paths()[0] / name, dpi=130)
    plt.close(fig)


def fig_curves(preds_sw, y_all, kind, models):
    fig, axes = model_grid(models)
    for ax, clf in zip(axes.flat, models):
        for fs in FEATURE_SETS:
            p = preds_sw[(clf, fs)]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y_all, p)
                ax.plot(fpr, tpr, color=COLORS[fs], label=f"{fs} (AUC {roc_auc_score(y_all, p):.3f})")
            else:
                precision, recall, _ = precision_recall_curve(1 - y_all, 1 - p)
                ax.plot(recall, precision, color=COLORS[fs],
                        label=f"{fs} (AP {average_precision_score(1 - y_all, 1 - p):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ":", color="#999")
            ax.set_xlabel("доля wake, названная сном")
            ax.set_ylabel("доля сна, названная сном")
        else:
            ax.axhline(np.mean(y_all == 0), ls=":", color="#999")
            ax.set_xlabel("доля wake, названная wake (recall)")
            ax.set_ylabel("доля верных среди названных wake (precision)")
        ax.set_title(clf, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right" if kind == "roc" else "upper right")
        style(ax)
    fig.suptitle("Рис. 2 (аналог): ROC сон/бодрствование, LOSO" if kind == "roc" else
                 "Рис. 3 (аналог): precision–recall для wake, LOSO", fontsize=14)
    save(fig, "fig2_roc_sleep_wake.png" if kind == "roc" else "fig3_pr_wake.png")


def fig4_three_class(curves, models):
    fig, axes = model_grid(models)
    for ax, clf in zip(axes.flat, models):
        for fs in FEATURE_SETS:
            c = curves[(clf, fs)]
            ax.plot(c.wake_fpr, (c.nrem_correct + c.rem_correct) / 2, color=COLORS[fs], label=fs)
            if fs == "Motion only":  # как в статье: для одной motion NREM и REM отдельно
                ax.plot(c.wake_fpr, c.nrem_correct, ":", color=COLORS[fs], label="Motion only: NREM")
                ax.plot(c.wake_fpr, c.rem_correct, "--", color=COLORS[fs], label="Motion only: REM")
        ax.set_title(clf, fontweight="bold")
        ax.set_xlabel("доля wake, названная REM или NREM")
        ax.set_ylabel("доля верных NREM и REM (уравненных)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
        style(ax)
    fig.suptitle("Рис. 4 (аналог): wake / NREM / REM, LOSO", fontsize=14)
    save(fig, "fig4_three_class.png")


def fig5_bland_altman(ba_all, ba_stats, model, suffix):
    titles = {"TST": "TST, мин", "SOL": "SOL, мин", "WASO": "WASO, мин", "SE": "SE (доля)", "REM": "REM, мин",
              "NREM": "NREM, мин"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for ax, metric in zip(axes.flat, titles):
        for fs in FEATURE_SETS:
            d = ba_all[(ba_all.feature_set == fs) & (ba_all.metric == metric)]
            ax.scatter(d.psg, d["diff"], s=18 if fs == "Motion, HR, Clock" else 10, color=COLORS[fs], alpha=.85,
                       label=fs)
        st = ba_stats[(ba_stats.feature_set == "Motion, HR, Clock") & (ba_stats.metric == metric)].iloc[0]
        ax.axhline(0, color="#999", lw=.8)
        ax.axhline(st.bias, color=COLORS["Motion, HR, Clock"], lw=1.2)
        for lim in (st.loa_low, st.loa_high):
            ax.axhline(lim, color=COLORS["Motion, HR, Clock"], lw=1, ls="--")
        ax.set_title(f"{titles[metric]}: сдвиг {st.bias:.2f}, границы [{st.loa_low:.2f}; {st.loa_high:.2f}]",
                     fontsize=10)
        ax.set_xlabel(f"PSG, {titles[metric]}")
        ax.set_ylabel("PSG − модель")
        style(ax)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Рис. 5 (аналог): Bland–Altman, {model}, θW = 0.3, θREM = 0.35; линии — сдвиг и ±1.96 SD "
                 "для Motion, HR, Clock", fontsize=12)
    save(fig, f"fig5_bland_altman{suffix}.png")


def fig6_7_per_subject(per6, per7, model, suffix):
    cols = [("accuracy", "accuracy"), ("wake_correct", "wake найден"), ("sleep_correct", "сон найден"), ("kappa", "κ"),
            ("f1_wake", "F1 wake")]
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 3.6))
    for ax, (col, title) in zip(axes, cols):
        values = per6[col].dropna()
        ax.hist(values, bins=np.linspace(min(0, values.min()), 1, 21), color=MODEL_COLORS[model], edgecolor="white")
        ax.axvline(values.median(), color="k", ls="--", lw=1)
        ax.set_title(f"{title}: медиана {values.median():.2f}", fontsize=10)
        ax.set_xlabel("по людям")
        style(ax)
    axes[0].set_ylabel("людей")
    fig.suptitle(f"Рис. 6 (аналог): качество по людям, {model}, Motion, HR, Clock, P(wake) ≥ 0.3 → wake", fontsize=12)
    save(fig, f"fig6_per_subject_theta{suffix}.png")

    fig, axes = plt.subplots(len(SENSITIVITIES), 2, figsize=(10, 11), sharex=True)
    for i, tpr in enumerate(SENSITIVITIES):
        d = per7[per7.tpr == tpr]
        for j, (col, title) in enumerate([("accuracy", "accuracy"), ("wake_correct", "wake найден (specificity)")]):
            ax = axes[i, j]
            ax.hist(d[col].dropna(), bins=np.linspace(0, 1, 21), color=MODEL_COLORS[model], edgecolor="white")
            ax.axvline(d[col].median(), color="k", ls="--", lw=1)
            ax.set_title(f"сон найден = {tpr:.2f}: {title}, медиана {d[col].median():.2f}", fontsize=9)
            style(ax)
    fig.suptitle(f"Рис. 7 (аналог), {model}: порог подобран для каждого человека под заданную долю сна", fontsize=12)
    save(fig, f"fig7_per_subject_tpr{suffix}.png")


def grouped_bars(ax, frame, col, models, title, ylim=None):
    width = 0.8 / len(FEATURE_SETS)
    x = np.arange(len(models))
    for i, fs in enumerate(FEATURE_SETS):
        vals = [frame[(frame.classifier == c) & (frame.feature_set == fs)][col].iloc[0] for c in models]
        bars = ax.bar(x + (i - (len(FEATURE_SETS) - 1) / 2) * width, vals, width, color=COLORS[fs], label=fs)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x, [c.replace(" ", "\n", 1) for c in models], fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    style(ax)


def fig_summary(sw, models):
    """Сводка метрик статьи: AUC, κ и F1 wake (сон найден = 0.9) для всех моделей и наборов признаков."""
    d = sw[sw.sensitivity_target == 0.9]
    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 4.8))
    grouped_bars(axes[0], d, "auc", models, "AUC", (0.5, 0.95))
    grouped_bars(axes[1], d, "kappa", models, "κ при 90% сна")
    grouped_bars(axes[2], d, "f1_wake", models, "F1 wake при 90% сна")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Все модели и наборы признаков, LOSO", fontsize=13)
    save(fig, "fig_summary_models.png")


def fig_new_metrics(sw, tc, models):
    """Метрики, которых нет в статье: AP, PPV, macro F1 и F1 REM."""
    d = sw[sw.sensitivity_target == 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    grouped_bars(axes[0, 0], d, "ap_wake", models, "AP wake (площадь под PR-кривой)", (0, 1))
    grouped_bars(axes[0, 1], d, "ppv_wake", models, "PPV wake: доля верных среди названных wake (90% сна)", (0, 1))
    grouped_bars(axes[1, 0], tc, "f1_macro_balanced", models, "3 класса: macro F1 (NREM = REM)", (0, 1))
    grouped_bars(axes[1, 1], tc, "f1_rem_balanced", models, "3 класса: F1 REM (NREM = REM)", (0, 1))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Новые метрики (в статье их нет), LOSO", fontsize=13)
    save(fig, "fig_new_metrics.png")


def fig_f1_threshold_models(preds_sw, y_all, models):
    """F1 бодрствования в зависимости от порога на P(wake) для всех моделей (Motion, HR, Clock)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for clf in models:
        p_wake = 1 - preds_sw[(clf, "Motion, HR, Clock")]
        precision, recall, thresholds = precision_recall_curve(1 - y_all, p_wake)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        i = int(np.argmax(f1[:-1]))
        ax.plot(thresholds, f1[:-1], color=MODEL_COLORS[clf], label=f"{clf}: max {f1[i]:.2f} при {thresholds[i]:.2f}")
        ax.scatter([thresholds[i]], [f1[i]], color=MODEL_COLORS[clf], zorder=3)
    ax.set_xlabel("порог на P(wake)")
    ax.set_ylabel("F1 wake")
    ax.set_title("F1 бодрствования в зависимости от порога (Motion, HR, Clock, LOSO)", fontweight="bold")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "fig_f1_threshold_models.png")


def feature_bars(ax, frame, col, models, title, ylabel, digits=3):
    """Столбцы: по оси x — признаки motion, hr, clock, цвет — модель."""
    width = 0.8 / len(models)
    x = np.arange(len(SHAPLEY_FEATURES))
    for i, clf in enumerate(models):
        vals = frame[frame.classifier == clf].set_index("feature").loc[SHAPLEY_FEATURES, col].values
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, color=MODEL_COLORS[clf], label=clf)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.{digits}f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=6)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x, SHAPLEY_FEATURES)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    style(ax)


def fig_shapley_auc(subsets, values, models):
    order = list(ALL_SETS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    height = 0.8 / len(models)
    y = np.arange(len(order))
    for i, clf in enumerate(models):
        auc = subsets[subsets.classifier == clf].set_index("feature_set").loc[order, "auc"].values
        axes[0].barh(y + (i - (len(models) - 1) / 2) * height, auc, height, color=MODEL_COLORS[clf], label=clf)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.5, min(1.0, subsets.auc.max() + 0.03))
    axes[0].set_xlabel("AUC сон/бодрствование (LOSO)")
    axes[0].set_title("Качество на всех 7 подмножествах признаков", fontweight="bold", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    style(axes[0])
    feature_bars(axes[1], values[values.metric == "auc"], "shapley", models,
                 "Значение Шепли: вклад признака в (AUC − 0.5)", "вклад в AUC")
    fig.suptitle("Важность признаков по Шепли: модели переобучены на каждом подмножестве признаков, LOSO", fontsize=13)
    save(fig, "fig_shapley_auc.png")


def fig_shapley_metrics(values, models):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (metric, name) in zip(axes.flat, SHAPLEY_METRICS.items()):
        d = values[values.metric == metric]
        feature_bars(ax, d, "shapley", models, f"{name}: случайная модель {d.baseline.iloc[0]:.3f}",
                     f"вклад в {name}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Значения Шепли по разным метрикам (сумма по признакам = качество полной модели − случайной)",
                 fontsize=13)
    save(fig, "fig_shapley_metrics.png")


def fig_shapley_share(values, models):
    """Доли значений Шепли (в % от прироста полной модели): модели × (метрика, признак)."""
    cols = [(m, f) for m in SHAPLEY_METRICS for f in SHAPLEY_FEATURES]
    grid = np.array([[values[(values.classifier == c) & (values.metric == m) & (values.feature == f)].share.iloc[0]
                      for m, f in cols] for c in models]) * 100
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.7 * len(models)))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), [f"{SHAPLEY_METRICS[m].split(' ')[0]}\n{f}" for m, f in cols], fontsize=8)
    ax.set_yticks(range(len(models)), models)
    for j in range(len(SHAPLEY_FEATURES), len(cols), len(SHAPLEY_FEATURES)):
        ax.axvline(j - 0.5, color="k", lw=1)
    plt.colorbar(im, ax=ax, label="доля прироста, %")
    ax.set_title("Доля каждого признака в приросте качества над случайной моделью (Шепли)", fontweight="bold")
    save(fig, "fig_shapley_share.png")


def fig_shap_beeswarm(shap_values, models):
    fig, axes = model_grid(models)
    rng = np.random.default_rng(0)
    for ax, clf in zip(axes.flat, models):
        values, X = shap_values[clf]
        for j, f in enumerate(SHAPLEY_FEATURES):
            rank = pd.Series(X[:, j]).rank(pct=True).values  # цвет — перцентиль значения признака
            sc = ax.scatter(values[:, j], j + rng.uniform(-0.3, 0.3, len(values)), c=rank, cmap="coolwarm", s=3,
                            alpha=.6, vmin=0, vmax=1)
        ax.axvline(0, color="k", lw=.6)
        ax.set_yticks(range(len(SHAPLEY_FEATURES)),
                      [f"{f}\n|SHAP| {np.abs(values[:, j]).mean():.3f}" for j, f in enumerate(SHAPLEY_FEATURES)])
        ax.invert_yaxis()
        ax.set_xlabel("SHAP: сдвиг P(сон) (> 0 — к сну, < 0 — к бодрствованию)")
        ax.set_title(clf, fontweight="bold")
        style(ax)
        plt.colorbar(sc, ax=ax, label="значение признака (перцентиль)")
    fig.suptitle("SHAP-значения по эпохам (Motion, HR, Clock, out-of-fold): синий — малое значение признака, "
                 "красный — большое", fontsize=13)
    save(fig, "fig_shap_beeswarm_models.png")


def fig_shap_mean_abs(summary, models):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    feature_bars(axes[0], summary, "mean_abs", models, "mean |SHAP|: средний сдвиг P(сон) от признака", "|SHAP|")
    feature_bars(axes[1], summary.assign(share=summary.share * 100), "share", models,
                 "доля признака в сумме mean |SHAP|, %", "%", digits=0)
    axes[0].legend(fontsize=8)
    fig.suptitle("Важность признаков по SHAP (Motion, HR, Clock, сон/бодрствование, LOSO)", fontsize=13)
    save(fig, "fig_shap_mean_abs.png")


def fig_shap_dependence(shap_values, models):
    color_by = {"motion": "clock", "hr": "clock", "clock": "motion"}
    fig, axes = plt.subplots(len(models), 3, figsize=(16, 3.4 * len(models)), squeeze=False)
    for i, clf in enumerate(models):
        values, X = shap_values[clf]
        for j, f in enumerate(SHAPLEY_FEATURES):
            ax = axes[i, j]
            k = SHAPLEY_FEATURES.index(color_by[f])
            sc = ax.scatter(X[:, j], values[:, j], c=X[:, k], s=3, cmap="coolwarm", alpha=.6,
                            vmin=np.quantile(X[:, k], 0.05), vmax=np.quantile(X[:, k], 0.95))
            ax.axhline(0, color="k", lw=.6)
            if f == "motion":
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(f)
            ax.set_ylabel(f"SHAP({f})")
            ax.set_title(f"{clf}: {f}", fontsize=10, fontweight="bold")
            style(ax)
            plt.colorbar(sc, ax=ax, label=color_by[f])
    fig.suptitle("SHAP dependence: как значение признака сдвигает P(сон) (> 0 — к сну)", fontsize=13)
    save(fig, "fig_shap_dependence_models.png")


# --------------------------------------------------------------------------- main
def main():
    global SMOKE
    parser = argparse.ArgumentParser(description="Все метрики статьи + F1 для моделей статьи и CatBoost (LOSO)")
    parser.add_argument("--models", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS,
                        help="какие модели обучать (по умолчанию все); в отчёт идут все уже посчитанные")
    parser.add_argument("--skip-nn", action="store_true", help="не обучать нейросеть")
    parser.add_argument("--plots-only", action="store_true",
                        help="ничего не обучать: таблицы и рисунки из сохранённых предсказаний")
    parser.add_argument("--no-report", action="store_true",
                        help="не собирать outputs/RESULTS.md в конце (run_all.py собирает его сам)")
    parser.add_argument("--smoke", action="store_true",
                        help="проверка, что всё запускается: 6 человек, крошечные модели, результаты в "
                             "outputs/repro/smoke/ (настоящие результаты не трогаются)")
    args = parser.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        sl.SUBJECTS[:] = sl.SUBJECTS[:6]
    out, cache = paths()
    cache.mkdir(parents=True, exist_ok=True)
    started = time.time()

    to_train = [] if args.plots_only else [m for m in args.models if not (args.skip_nn and m == "Neural Net")]
    if "CatBoost" in to_train and not catboost_available():
        print('CatBoost пропущен: пакет не установлен (pip install "catboost>=1.2" или python repro/run_all.py)')
        to_train.remove("CatBoost")

    data = sl.load_paper_features()
    y_sw = {s: sl.sleep_wake(data[s]["stage"]) for s in sl.SUBJECTS}
    y_3 = {s: sl.three_class(data[s]["stage"]) for s in sl.SUBJECTS}
    Y_sw = np.concatenate([y_sw[s] for s in sl.SUBJECTS])
    Y_3 = np.concatenate([y_3[s] for s in sl.SUBJECTS])

    per_t1, sum_t1, t1_source = table1(data)
    per_t1.to_csv(out / "table1_per_subject.csv", index=False, float_format="%.2f")
    sum_t1.to_csv(out / "table1_summary.csv", index=False, float_format="%.2f")
    (out / "table1_source.txt").write_text(t1_source, encoding="utf-8")

    # 1. обучение: этапы «модель → признаки → задача», прогресс по фолдам
    # для каждой модели: 4 набора статьи × 2 задачи, ещё 3 подмножества для Шепли (сон/бодрствование) и этап SHAP
    todo = []
    for clf in to_train:
        todo += [(clf, fs, t) for fs in ALL_SETS for t in TARGETS
                 if (fs in FEATURE_SETS or t == "sw") and not cache_path(clf, fs, t).exists()]
        if not shap_path(clf).exists():
            todo.append((clf, SHAP_STAGE, "sw"))
    if todo:
        models_todo = list(dict.fromkeys(clf for clf, _, _ in todo))
        bar = Progress(len(todo) * len(sl.SUBJECTS), "обучение")
        bar.message(f"Обучение: {len(models_todo)} моделей ({', '.join(models_todo)}), {len(todo)} этапов по "
                    f"{len(sl.SUBJECTS)} фолдов, {N_JOBS} процессов. Уже посчитанное берётся из сохранённого.")
        for step, (clf, fs, target) in enumerate(todo, 1):
            k = models_todo.index(clf) + 1
            if (clf, fs, target) == next(x for x in todo if x[0] == clf):
                bar.message(f"== модель {k}/{len(models_todo)}: {clf}")
            t0 = time.time()
            if fs == SHAP_STAGE:
                status = f"этап {step}/{len(todo)}: {clf} | SHAP-значения (Motion, HR, Clock)"
                shap_loso(data, clf, bar, status)
                bar.message(f"   готово: SHAP-значения — {fmt_time(time.time() - t0)}")
                continue
            status = f"этап {step}/{len(todo)}: {clf} | {fs} | {TARGETS[target]}"
            loso_predictions(data, clf, fs, target, bar, status)
            bar.message(f"   готово: {fs}, {TARGETS[target]} — {fmt_time(time.time() - t0)}")
        bar.close()
    elif not args.plots_only:
        print("Все выбранные модели уже посчитаны — беру сохранённые предсказания "
              f"(чтобы переобучить, удалите {cache.relative_to(sl.ROOT)})")

    # 2. метрики и рисунки: все модели, чьи предсказания сохранены полностью
    models = [c for c in CLASSIFIERS if cached(c)]
    if not models:
        sys.exit("Нет сохранённых предсказаний: запустите без --plots-only")
    main_model = "Neural Net" if "Neural Net" in models else ("Random Forest" if "Random Forest" in models
                                                              else models[-1])
    print(f"\nТаблицы и рисунки: модели {', '.join(models)}; рис. 5–7 без суффикса — по модели {main_model}", flush=True)
    (out / "models_included.txt").write_text("\n".join(models), encoding="utf-8")
    (out / "per_subject_model.txt").write_text(main_model, encoding="utf-8")

    figure_bar = Progress(len(models) * len(FEATURE_SETS) + len(models) + 2, "таблицы и рисунки")
    sw_rows, tc_rows, curves, preds_sw = [], [], {}, {}
    ba_frames, ba_stats_frames, per6_frames, per7_frames = [], [], [], []
    for clf in models:
        ba_model = []
        for fs in FEATURE_SETS:
            p_sw = loso_predictions(data, clf, fs, "sw")
            p_3 = loso_predictions(data, clf, fs, "3")
            P_sw = np.concatenate([p_sw[s][:, 1] for s in sl.SUBJECTS])
            P_3 = np.vstack([p_3[s] for s in sl.SUBJECTS])
            preds_sw[(clf, fs)] = P_sw
            for r in sleep_wake_metrics(Y_sw, P_sw):
                sw_rows.append(dict(classifier=clf, feature_set=fs, **r))
            r6 = sl.three_class_at_wake_fpr(Y_3, P_3, 0.4)
            fixed = sl.three_class_f1(Y_3, sl.fixed_threshold_predict(P_3, THETA_WAKE, THETA_REM))
            tc_rows.append(dict(classifier=clf, feature_set=fs, wake_correct=r6["wake_correct"],
                                nrem_correct=r6["nrem_correct"], rem_correct=r6["rem_correct"],
                                best_accuracy=r6["best_accuracy"], kappa=r6["best_kappa"],
                                f1_macro_best=r6["best_f1_macro"], f1_macro_balanced=r6["balanced_f1_macro"],
                                f1_rem_balanced=r6["balanced_f1_rem"], f1_wake_fixed=fixed["f1_wake"],
                                f1_nrem_fixed=fixed["f1_nrem"], f1_rem_fixed=fixed["f1_rem"],
                                f1_macro_fixed=fixed["f1_macro"]))
            curves[(clf, fs)] = pd.DataFrame([sl.three_class_at_wake_fpr(Y_3, P_3, f)
                                              for f in np.linspace(0.05, 0.95, 19)])
            ba = bland_altman(p_3, y_3).assign(classifier=clf, feature_set=fs)
            ba_model.append(ba)
            if fs == "Motion, HR, Clock":
                per6, per7 = per_subject(p_sw, y_sw)
                per6_frames.append(per6.assign(classifier=clf))
                per7_frames.append(per7.assign(classifier=clf))
            figure_bar.update(status=f"{clf} | {fs}")
        ba_model = pd.concat(ba_model)
        stats = ba_model.groupby(["feature_set", "metric"], sort=False)["diff"].agg(["mean", "std"]).reset_index()
        stats.columns = ["feature_set", "metric", "bias", "sd"]
        stats["loa_low"], stats["loa_high"] = stats.bias - 1.96 * stats.sd, stats.bias + 1.96 * stats.sd
        stats.insert(0, "classifier", clf)
        ba_frames.append(ba_model)
        ba_stats_frames.append(stats)
        # рисунки 5–7 для каждой модели; без суффикса — для основной (как в статье)
        for suffix in ([f"__{slug(clf)}"] + ([""] if clf == main_model else [])):
            fig5_bland_altman(ba_model, stats, clf, suffix)
            fig6_7_per_subject(per6_frames[-1], per7_frames[-1], clf, suffix)
        figure_bar.update(status=f"{clf} | рисунки 5–7")

    sw, tc = pd.DataFrame(sw_rows), pd.DataFrame(tc_rows)
    sw.to_csv(out / "sleep_wake_all_models.csv", index=False, float_format="%.4f")
    tc.to_csv(out / "three_class_all_models.csv", index=False, float_format="%.4f")
    pd.concat(ba_frames).to_csv(out / "bland_altman_per_subject.csv", index=False, float_format="%.4f")
    pd.concat(ba_stats_frames).to_csv(out / "bland_altman_stats.csv", index=False, float_format="%.4f")
    pd.concat(per6_frames).to_csv(out / "per_subject_theta.csv", index=False, float_format="%.4f")
    pd.concat(per7_frames).to_csv(out / "per_subject_tpr.csv", index=False, float_format="%.4f")

    fig_curves(preds_sw, Y_sw, "roc", models)
    fig_curves(preds_sw, Y_sw, "pr", models)
    fig4_three_class(curves, models)
    fig_summary(sw, models)
    fig_new_metrics(sw, tc, models)
    fig_f1_threshold_models(preds_sw, Y_sw, models)
    figure_bar.update(status="общие рисунки")

    # важность признаков по Шепли: модели, у которых посчитаны все 7 подмножеств признаков / SHAP-значения
    shapley_models = [c for c in models if shapley_cached(c)]
    if shapley_models:
        subsets, values = shapley_tables(data, shapley_models, Y_sw)
        subsets.to_csv(out / "shapley_subsets.csv", index=False, float_format="%.4f")
        values.to_csv(out / "shapley_values.csv", index=False, float_format="%.4f")
        fig_shapley_auc(subsets, values, shapley_models)
        fig_shapley_metrics(values, shapley_models)
        fig_shapley_share(values, shapley_models)
    shap_models = [c for c in models if shap_path(c).exists()]
    if shap_models:
        shap_values = {c: shap_loso(data, c) for c in shap_models}
        shap_summary(shap_values).to_csv(out / "shap_mean_abs.csv", index=False, float_format="%.4f")
        fig_shap_beeswarm(shap_values, shap_models)
        fig_shap_mean_abs(shap_summary(shap_values), shap_models)
        fig_shap_dependence(shap_values, shap_models)
    figure_bar.update(status="важность признаков (Шепли, SHAP)")
    figure_bar.close()

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(sw[sw.sensitivity_target == 0.9][["classifier", "feature_set", "accuracy", "wake_correct", "kappa", "auc",
                                              "f1_wake", "f1_macro"]].round(3).to_string(index=False))
    no_shapley = [m for m in models if not (shapley_cached(m) and shap_path(m).exists())]
    if no_shapley:
        print(f"\nВажность по Шепли ещё не посчитана для: {', '.join(no_shapley)}. Досчитать (обучит только "
              f"недостающее): python repro/09_paper_metrics.py --models " + " ".join(f'"{m}"' for m in no_shapley))
    missing = [m for m in CLASSIFIERS if m not in models]
    if missing:
        print(f"\nЕщё не обучены: {', '.join(missing)}. Дообучить (остальное возьмётся из сохранённого): "
              f"python repro/09_paper_metrics.py --models " + " ".join(f'"{m}"' for m in missing))
    print(f"\nГотово за {fmt_time(time.time() - started)}. Таблицы и рисунки: {out.relative_to(sl.ROOT)}", flush=True)
    if not args.no_report:  # всё — в один Markdown-файл
        command = [sys.executable, str(REPRO / "08_results_markdown.py")]
        if SMOKE:
            command += ["--analysis-dir", str(out.parent), "--out", str(out.parent / "RESULTS.md")]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
