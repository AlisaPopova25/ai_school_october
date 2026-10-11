"""ВСЕ настройки обучения CatBoost — в этом файле. Менять только здесь.

FAST = True   — быстрый режим: простые параметры, без перебора, мало разбиений. Минуты; чтобы проверить,
                что всё запускается и рисуется.
FAST = False  — полный протокол статьи: до 3000 деревьев с ранней остановкой, перебор 12 сочетаний,
                50 / 20 разбиений как в статье. Часы; для итоговых результатов.

Что откуда берётся:
  PARAMS              -> source/utils.py (get_catboost_classifiers): параметры модели
  GRID                -> source/analysis/classification/parameter_search.py: перебор GridSearchCV
                         (одно значение в каждом списке = перебора нет, параметры берутся как есть)
  SW_TRIALS,
  THREE_CLASS_TRIALS  -> repro/train_catboost.py: сколько случайных разбиений по людям
                         (их можно переопределить флагами --sw-trials / --three-class-trials)
"""

FAST = True

if FAST:
    PARAMS = dict(
        iterations=100,              # деревьев (мало — быстро)
        learning_rate=0.1,           # шаг обучения (крупный, раз деревьев мало)
        depth=4,                     # глубина дерева
        l2_leaf_reg=3.0,             # L2-регуляризация листьев
        early_stopping_rounds=None,  # без ранней остановки
        refit_on_full=False,         # без повторного обучения на всех данных
        random_seed=0,
    )
    GRID = {"depth": [4]}            # перебора нет: одно сочетание
    SW_TRIALS = 10                   # разбиений для сна/бодрствования (в статье 50)
    THREE_CLASS_TRIALS = 5           # разбиений для трёх классов (в статье 20)
else:
    PARAMS = dict(
        iterations=3000,             # максимум деревьев; реально — сколько выберет ранняя остановка
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=200,   # остановка, если 200 деревьев подряд нет улучшения
        eval_fraction=0.15,          # доля обучающих данных для ранней остановки
        refit_on_full=True,          # затем переобучение на всех данных с найденным числом деревьев
        random_seed=0,
    )
    GRID = {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "l2_leaf_reg": [3, 10]}  # 12 сочетаний
    SW_TRIALS = 50
    THREE_CLASS_TRIALS = 20
