# Предсказание стадий сна по Apple Watch

Воспроизведение и развитие статьи Walch O., Huang Y., Forger D., Goldstein C.
*Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer
wearable device*. SLEEP 42(12), zsz180 (2019). [doi:10.1093/sleep/zsz180](https://doi.org/10.1093/sleep/zsz180)

По сырому ускорению и пульсу с Apple Watch модель определяет для каждой 30-секундной эпохи сон или бодрствование
(и wake / NREM / REM); эталон — полисомнография (PSG) у 31 человека. Мы запустили оригинальный код на
современном Python, воспроизвели рисунки и таблицы статьи, разобрали модель и признаки и проверили, как её улучшить.

## Структура

```
├── source/            код статьи (ojwalch/sleep_classifiers, MIT) с нашими правками
│   ├── preprocessing/ сырые сигналы → признаки motion, hr, clock (clock_proxy/ — наш Python-порт модели часов)
│   └── analysis/      классификаторы, кросс-валидация, рисунки и таблицы статьи (+ CatBoost)
├── repro/             наши скрипты: 01–06 анализ, sleeplib.py, ext_features.py, загрузка данных,
│                      запуск CatBoost, пошаговый ноутбук walkthrough.ipynb
├── data/              сюда скачивается датасет PhysioNet sleep-accel (ссылка и скрипт — data/README.md)
├── outputs/           куда пишет код; в git — готовые признаки (outputs/features, outputs/repro/ext_features)
├── results/           снимок результатов: рисунки статьи, таблицы, наш анализ (см. results/README.md)
├── docs/              DATASET_RU.md — данные и признаки; LOGREG_RU.md — модель, результаты, вопросы к защите
├── paper/             статья (PDF, CC BY-NC 4.0) и извлечённый текст
├── archive/           старые скрипты авторов (не используются)
└── .vscode/, .env     конфигурации запуска для VS Code
```

## Быстрый старт

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # точные проверенные версии — requirements-lock.txt
```

Все анализы работают сразу на готовых признаках из репозитория:

```bash
export PYTHONPATH=".:repro"       # PowerShell: $env:PYTHONPATH = ".;repro"
python repro/02_run_original_analysis.py mc_sw --only "Logistic Regression"   # рис. 2–3, табл. 2 (~1 мин)
python repro/02_run_original_analysis.py mc_3class --only "Logistic Regression"  # рис. 4, S1–S3, табл. 6
python repro/05_logreg_deep_dive.py      # разбор логистической регрессии
python repro/03_feature_importance.py    # важность признаков, SHAP
python repro/04_improve_model.py         # расширенные признаки, LightGBM, HMM
python repro/06_quality_by_time.py       # качество по часам ночи
```

Пересчитать признаки с нуля и рисунок 1 — нужен датасет (2.3 ГБ, в репозитории не хранится,
[PhysioNet sleep-accel](https://physionet.org/content/sleep-accel/1.0.0/)):

```bash
python repro/download_data.py                          # скачивает датасет в data/, сверяет SHA256
python source/preprocessing/preprocessing_runner.py    # ~3 мин
```

В VS Code: откройте папку, выберите интерпретатор `.venv`, затем **Run and Debug** — готовые конфигурации
`1`–`5` (переменные окружения берутся из `.env`; разделитель путей в нём виндовый `;`).

Полезные переменные: `SLEEP_N_JOBS` — число процессов (каждый ~1.2 ГБ памяти), `SLEEP_INCLUDE_CATBOOST=0` —
отключить CatBoost. Обучение CatBoost на мощной машине — `repro/run_catboost.ps1` / `repro/run_catboost.sh`.

## Ключевые результаты

**Воспроизведение (логистическая регрессия, оригинальный код, 50 разбиений по людям):**

| признаки | AUC: статья / мы | доля распознанного бодрствования при 90% сна: статья / мы |
|---|---|---|
| Motion | 0.819 / 0.838 | 0.549 / 0.537 |
| HR | 0.743 / 0.714 | 0.326 / 0.305 |
| Motion, HR | 0.830 / 0.836 | 0.546 / 0.537 |
| Motion, HR, Clock | 0.854 / 0.855 | 0.556 / 0.633 |

Расхождение в последней строке — из-за clock: в открытых данных нет времени суток, и для модели часов принято
допущение «запись начинается в 23:00».

**Что выяснили сверх статьи** (leave-one-subject-out):

- Вклад признаков в AUC логистической регрессии (значения Шепли по всем подмножествам): motion 44%, hr 29%, clock 27%.
- Clock один даёт AUC 0.675 и в основном отмечает начало ночи (засыпание); простой косинус от начала записи
  почти так же хорош (0.853 против 0.855).
- Качество зависит от времени ночи: после 7 часов AUC падает до 0.69, а clock сдвигает ответы модели
  («рано — бодрствование», «поздно — REM»).
- Около 50 расширенных признаков (многомасштабные окна активности, угол запястья, время с последнего движения,
  уровень пульса) поднимают ту же логистическую регрессию с AUC 0.853 до **0.916** — почти как LightGBM (0.919);
  для трёх классов κ растёт с 0.34 до 0.43. Качество ограничивают признаки, а не модель.

Подробно — [docs/LOGREG_RU.md](docs/LOGREG_RU.md), данные — [docs/DATASET_RU.md](docs/DATASET_RU.md).

## Гипотезы для второй половины проекта

| | гипотеза | данные |
|---|---|---|
| H1 | важности признаков для REM и NREM различаются (SHAP по классам с бутстреп-интервалами) | наши |
| H2 | есть оптимальная длина окна признаков; окно «только прошлое» почти не хуже | наши |
| H3 | качество зависит от времени ночи — **подтверждена** | наши |
| H4 | обучение на похожем датасете улучшает метрики | [BIDSleep](https://physionet.org/content/bidsleep-dataset/1.0.0/) (Apple Watch, 47 человек, 253 ночи) |
| H5 | модель одинаково обобщается на мужчин и женщин | Newcastle, MESA, DREAMT |
| H6 | качество падает для старших возрастов | MESA, DREAMT, Newcastle |
| H7 | низкий κ трёх классов — из-за редкого пульса (гипотеза авторов) | наши, прореживание пульса |
| H8 | модель часов с реальным временем суток лучше косинуса | BIDSleep |
| H9 | учёт соседних эпох (HMM) улучшает REM/NREM — первый результат: κ 0.23 → 0.37 | наши |

В открытом датасете нет пола и возраста по людям, поэтому H5–H6 требуют других данных.

## Что изменено в коде статьи

Python-порт MATLAB-модели циркадных часов; совместимость с Python 3.12 и scikit-learn 1.9 (в том числе
one-vs-rest логистическая регрессия как в scikit-learn 0.20); фиксированный seed разбиений; закрытие пула процессов
и ограничение их числа; CatBoost как пятая модель; сетка общих рисунков под любое число моделей. Найденные
методологические ограничения кода (нестандартизованные признаки, пороги по тестовой разметке и др.) описаны в
docs/LOGREG_RU.md, §5. История изменений — отдельными коммитами поверх неизменённого оригинала.

## Источники и лицензии

- Статья: © Sleep Research Society 2019, открытый доступ по CC BY-NC 4.0.
- Код статьи: [ojwalch/sleep_classifiers](https://github.com/ojwalch/sleep_classifiers), MIT.
- Данные: [PhysioNet sleep-accel v1.0.0](https://physionet.org/content/sleep-accel/1.0.0/), Open Data Commons Attribution v1.0.
