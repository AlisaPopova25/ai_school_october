"""Запуск ОРИГИНАЛЬНОГО кода анализа статьи (source/analysis) с фиксированным seed и настраиваемым числом сплитов.

Использование (из корня репозитория, PYTHONPATH=.):
    python repro/02_run_original_analysis.py mc_sw      [trials=50]   # Рис. 2, 3; Таблицы 2–5
    python repro/02_run_original_analysis.py mc_3class  [trials=20]   # Рис. 4, S1–S3; Таблица 6
    python repro/02_run_original_analysis.py loo_sw                   # Рис. 6, 7 (гистограммы по людям)
    python repro/02_run_original_analysis.py loo_3class               # Рис. 5 (Bland–Altman)
    python repro/02_run_original_analysis.py time_features [trials=50]  # Рис. S4 (clock vs cosine vs time)
    python repro/02_run_original_analysis.py combine                  # собрать общие figure_*.png из готовых панелей

Опция --only "Logistic Regression" — запустить только один классификатор
(имена: "Random Forest", "Logistic Regression", "k-Nearest Neighbors", "Neural Net", "CatBoost").
CatBoost подключается автоматически, если установлен пакет catboost (pip install "catboost>=1.2");
SLEEP_INCLUDE_CATBOOST=0 его выключает. С ним общие картинки figure_*.png собираются сеткой 2x3.

Картинки -> outputs/figures/, таблицы (LaTeX, как в статье) -> stdout.
"""
import random
import sys
import time

from sklearn.neural_network import MLPClassifier

from source import utils
from source.analysis.classification.classifier_summary_builder import SleepWakeClassifierSummaryBuilder, \
    ThreeClassClassifierSummaryBuilder
from source.analysis.figures.curve_plot_builder import CurvePlotBuilder
from source.analysis.figures.performance_plot_builder import PerformancePlotBuilder
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType
from source.analysis.tables.table_builder import TableBuilder
from source.constants import Constants


ONLY = None


def selected_classifiers():
    return [c for c in utils.get_classifiers() if ONLY is None or c.name == ONLY]


def neural_net():
    return AttributedClassifier(name='Neural Net',
                                classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
                                                         max_iter=1000, alpha=0.01, solver='lbfgs'))


def mc_sw(trials):
    classifiers = selected_classifiers()
    for clf in classifiers:
        print('Running ' + clf.name + '...', flush=True)
        summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(clf, utils.get_base_feature_sets(), trials)
        CurvePlotBuilder.make_roc_sw(summary)
        CurvePlotBuilder.make_pr_sw(summary)
        TableBuilder.print_table_sw(summary)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trials, '_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trials, '_sw_roc')


def mc_3class(trials):
    classifiers = selected_classifiers()
    summaries = []
    for clf in classifiers:
        print('Running ' + clf.name + '...', flush=True)
        summary = ThreeClassClassifierSummaryBuilder.build_monte_carlo(clf, utils.get_base_feature_sets(), trials)
        CurvePlotBuilder.make_roc_one_vs_rest(summary)
        summary.performance_dictionary = CurvePlotBuilder.make_three_class_roc(summary)
        summaries.append(summary)
    TableBuilder.print_table_three_class(summaries)
    for ext in ['_three_class_roc', '_ovr_rem_roc', '_ovr_nrem_roc', '_ovr_wake_roc']:
        CurvePlotBuilder.combine_plots_as_grid(classifiers, trials, ext)


def loo_sw(_):
    summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(
        neural_net(), [[FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]])
    PerformancePlotBuilder.make_histogram_with_thresholds(summary)
    PerformancePlotBuilder.make_single_threshold_histograms(summary)


def loo_3class(_):
    summary = ThreeClassClassifierSummaryBuilder.build_leave_one_out(neural_net(), utils.get_base_feature_sets())
    PerformancePlotBuilder.make_bland_altman(summary)


def time_features(trials):
    classifiers = selected_classifiers()
    feature_sets = [[FeatureType.count, FeatureType.heart_rate],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.time],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.cosine],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]
    for clf in classifiers:
        print('Running ' + clf.name + '...', flush=True)
        summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(clf, feature_sets, trials)
        CurvePlotBuilder.make_roc_sw(summary, '_time_only')
        CurvePlotBuilder.make_pr_sw(summary, '_time_only')
        TableBuilder.print_table_sw(summary)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trials, '_time_only_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trials, '_time_only_sw_roc')


MODEL_ORDER = ['Random Forest', 'Logistic Regression', 'k-Nearest Neighbors', 'Neural Net', 'CatBoost']
COMBINED = [('_sw_roc', 50), ('_sw_pr', 50), ('_time_only_sw_roc', 50), ('_time_only_sw_pr', 50),
            ('_three_class_roc', 20), ('_ovr_rem_roc', 20), ('_ovr_nrem_roc', 20), ('_ovr_wake_roc', 20)]


def combine(_):
    """Собрать общие figure_*.png из уже готовых панелей моделей (например, после запусков с --only).

    Берутся все модели, чьи картинки лежат в outputs/figures; порядок — как в статье, CatBoost последним.
    """
    from types import SimpleNamespace
    for extension, trials in COMBINED:
        present = [SimpleNamespace(name=n) for n in MODEL_ORDER
                   if Constants.FIGURE_FILE_PATH.joinpath(f'{n}_{trials}_{extension}.png').exists()]
        if present:
            CurvePlotBuilder.combine_plots_as_grid(present, trials, extension)
            print(f'figure_{trials}{extension}.png: ' + ', '.join(m.name for m in present))


TASKS = {'mc_sw': (mc_sw, 50), 'mc_3class': (mc_3class, 20), 'loo_sw': (loo_sw, 0), 'loo_3class': (loo_3class, 0),
         'time_features': (time_features, 50), 'combine': (combine, 0)}

if __name__ == '__main__':
    args = sys.argv[1:]
    if '--only' in args:
        i = args.index('--only')
        ONLY = args[i + 1]
        del args[i:i + 2]
    task, default_trials = TASKS[args[0]]
    trials = int(args[1]) if len(args) > 1 else default_trials
    Constants.VERBOSE = False
    random.seed(2019)  # TrainTestSplitter.by_fraction uses random.shuffle
    start = time.time()
    task(trials)
    print(f'{args[0]} done in {(time.time() - start) / 60:.1f} min', flush=True)
