# Результаты (снимок)

Снимок результатов на октябрь 2026. При запуске код пишет свежие результаты в `outputs/` (картинки статьи —
в `outputs/figures/`, наш анализ — в `outputs/repro/`); эта папка их не перезаписывает.

## Воспроизведение статьи — `figures_paper/`, `tables/`

Оригинальный код статьи, Monte Carlo по людям 70/30, seed 2019 (`repro/02_run_original_analysis.py`).

| статья | файл |
|---|---|
| Рис. 1 (пример ночи) | `figures_paper/data_validation_<id>.png` — для всех 31 испытуемых |
| Рис. 2, ROC sleep/wake (50 разбиений) | `figures_paper/Logistic Regression_50__sw_roc.png`, `Random Forest_50__sw_roc.png`, общая `figure_50_sw_roc.png` |
| Рис. 3, precision–recall для wake | `figures_paper/*_50__sw_pr.png`, `figure_50_sw_pr.png` |
| Рис. 4, wake/NREM/REM (20 разбиений) | `figures_paper/Logistic Regression_20__three_class_roc.png` |
| Рис. S1–S3, one-vs-rest | `figures_paper/Logistic Regression_20__ovr_{rem,nrem,wake}_roc.png` |
| Рис. S4, clock / cosine / time | `figures_paper/Logistic Regression_50__time_only_sw_roc.png` |
| Таблица 2 (логистическая регрессия) | `tables/logreg_mc_sw.txt` (LaTeX, как в статье) |
| Таблица 6, строки LR | `tables/logreg_mc_3class.txt` |
| Сравнение clock / cosine / time | `tables/logreg_time_features.txt` |
| F1 к Таблицам 2 и 6 (добавлено, старые таблицы те же) | `tables/logreg_f1_mc_sw.txt`, `tables/logreg_f1_mc_3class.txt` — старая таблица, затем F1-таблица |

Готовы панели логистической регрессии (все) и случайного леса (рис. 2–3). kNN, нейросеть и CatBoost ещё не
прогнаны: общие картинки `figure_*.png` пока собраны из готовых панелей (`python repro/02_run_original_analysis.py combine`).

## Наш анализ

| папка | что внутри | скрипт |
|---|---|---|
| `dataset/` | сводка по испытуемым, покрытие сигналов, стадии, пример ночи и недели | `repro/01_explore_dataset.py` |
| `importance/` | Шепли по AUC на всех подмножествах признаков, веса моделей, permutation importance, SHAP (в т.ч. вдоль ночи и по классам) | `repro/03_feature_importance.py` |
| `improve/` | ~50 расширенных признаков, LightGBM, HMM-сглаживание, абляция групп признаков, SHAP | `repro/04_improve_model.py` |
| `logreg/` | логистическая регрессия: уравнение, бутстреп-интервалы, калибровка, путь регуляризации, проверка линейности, OvR vs softmax | `repro/05_logreg_deep_dive.py` |
| `time_of_night/` | качество по часам ночи, с clock и без | `repro/06_quality_by_time.py` |
| `f1/` | F1 бодрствования и macro F1 против accuracy (с моделью «всегда сон»), F1 от порога, F1 по стадиям и по людям | `repro/07_f1_metrics.py` |

Разбор и интерпретация — [docs/LOGREG_RU.md](../docs/LOGREG_RU.md), про F1 — [docs/F1_RU.md](../docs/F1_RU.md).
