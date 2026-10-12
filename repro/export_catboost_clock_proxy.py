"""Publish the already trained clock-proxy CatBoost models and LOSO predictions."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'repro'))
import sleeplib as sl

SOURCE = ROOT / 'results/catboost_fast_features'
OUT = ROOT / 'results/catboost_clock_proxy'
FEATURE_SETS = ['Motion, HR, Clock', 'Motion, HR, Clock + быстрые']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    manifest = dict(original)
    manifest['time_feature'] = 'clock_proxy'
    manifest['feature_sets'] = {name: original['feature_sets'][name] for name in FEATURE_SETS}
    manifest['models'] = {name: original['models'][name] for name in ['catboost_clock_base', 'catboost_clock_fast']}
    manifest['oof_sha256'] = {name: original['oof_sha256'][name] for name in FEATURE_SETS}
    manifest['publication'] = 'Existing trained weights and held-out predictions; no retraining or relabelling.'
    for stem, expected_hash in manifest['models'].items():
        model = SOURCE / f'{stem}.cbm'
        if hashlib.sha256(model.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Model differs from training manifest: {model}')
        shutil.copy2(model, OUT / model.name)
        info = json.loads((SOURCE / f'{stem}.json').read_text(encoding='utf-8'))
        info['time_feature'] = 'clock_proxy'
        if info['feature_names'][2] != 'clock':
            raise ValueError('Expected the actual clock-proxy input')
        (OUT / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ['participant_metrics.csv', 'participant_mean_metrics.csv', 'results.csv']:
        frame = pd.read_csv(SOURCE / name)
        frame = frame[frame.feature_set.isin(FEATURE_SETS)]
        if set(frame.feature_set) != set(FEATURE_SETS):
            raise ValueError(f'Missing clock-proxy rows: {name}')
        frame.to_csv(OUT / name, index=False)
    data = sl.load_paper_features(original['participants'])
    arrays = {
        'y': np.concatenate([sl.sleep_wake(data[s]['stage']) for s in original['participants']]),
        'subjects': np.concatenate([np.repeat(s, len(data[s]['stage'])) for s in original['participants']]),
    }
    for name in FEATURE_SETS:
        path = ROOT / 'outputs/repro/fast_features/oof' / f'catboost__{name}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['oof_sha256'][name]:
            raise ValueError(f'Predictions differ from training manifest: {name}')
        with np.load(path, allow_pickle=False) as stored:
            probabilities = [stored[s] for s in original['participants']]
            for s, p in zip(original['participants'], probabilities):
                if len(p) != len(data[s]['stage']) or not np.isfinite(p).all():
                    raise ValueError(f'Invalid predictions: {s}')
            arrays[name] = np.concatenate(probabilities)
    predictions = OUT / 'oof_predictions.npz'
    np.savez_compressed(predictions, **arrays)
    manifest['published_predictions_sha256'] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'README.md').write_text('''# Улучшенный CatBoost — clock proxy

Сравнение **Motion + HR + Clock proxy** и тех же признаков с тремя добавками: пиком движения, долей движения за минуту и относительным пульсом.

## Результаты

Среднее по 31 тестовому участнику, порог P(сон) = 0,5:

| Набор | ROC AUC | Cohen κ | Accuracy | Сон распознан | Бодрствование распознано |
|---|---:|---:|---:|---:|---:|
| Исходные 3 признака | 0,912 | 0,377 | 82,8% | 83,2% | 81,1% |
| Исходные + 3 быстрых | **0,922** | **0,409** | **85,1%** | **86,0%** | 80,7% |

Средний прирост: AUC +0,010, κ +0,032. κ улучшилась у 26 из 31 участника. При пороге 0,5 прирост в основном связан с распознаванием сна.

По объединённым тестовым эпохам: AUC 0,880 → 0,890; κ 0,357 → 0,389. Эти числа отличаются от средних по участникам, поскольку способ объединения другой.

[Автономный ноутбук с готовым анализом](catboost_clock_analysis.ipynb): редактируемые подписи, сравнение метрик, парные изменения по участникам, ROC/PR и матрицы ошибок. Обучение при открытии не запускается. [Графики и таблицы отдельно](analysis_exports).

## Как обучали

31 участник, 25 481 эпоха. 31 разбиение **leave-one-subject-out**: каждый участник тестируется моделью, обученной на остальных 30. База и расширенный вариант проверены на одних и тех же участниках.

Фиксированные параметры: 100 деревьев, глубина 4, learning_rate 0,1, l2_leaf_reg 3, random_seed 0, веса классов balanced. Расширенного поиска гиперпараметров нет. CatBoost 1.2.10, исходное обучение на CPU. Сглаживания предсказаний нет.

Clock proxy — готовый циркадный признак проекта, рассчитанный по истории активности. На вход подаются исходные значения `clock`; знак для модели не меняется. Быстрые признаки рассчитаны модулем [additional_sleep_features](../../additional_sleep_features/README.md). Минутное окно включает будущие измерения, медиана пульса берётся по всей записи: это анализ готовой записи.

## Веса и гиперпараметры

- [Улучшенная модель: 6 признаков](catboost_clock_fast.cbm), [порядок входов и параметры](catboost_clock_fast.json).
- [Исходная модель: 3 признака](catboost_clock_base.cbm), [порядок входов и параметры](catboost_clock_base.json).

Финальные веса обучены на всех 31 участниках. Приведённые оценки получены отдельными моделями внешних разбиений; финальные веса для оценки не использовались.

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("results/catboost_clock_proxy/catboost_clock_fast.cbm")
# X: столбцы строго в порядке motion, hr, clock, peak, active, hr_rel.
p_sleep = model.predict_proba(X)[:, 1]
sleep = p_sleep >= 0.5
```

Для повторного обучения только с Clock:

```bash
python repro/10_fast_features.py --models CatBoost --base Clock
```

Сборка анализа из опубликованных предсказаний:

```bash
python repro/build_catboost_clock_notebook.py
```

`oof_predictions.npz` содержит только отложенные предсказания двух вариантов, метки и идентификаторы участников. `manifest.json` сохраняет происхождение и SHA-256 весов, предсказаний и исходного кода обучения.

## Ограничения

Это исследовательская выборка, уже использованная для разных экспериментов. Описательные bootstrap-интервалы в ноутбуке не заменяют внешнюю проверку. Строки при чувствительности ко сну 90% используют порог, выбранный по тестовым меткам; основное сравнение выше использует фиксированный порог 0,5. С MLP, проверенным на пяти внешних разбиениях, напрямую сравнивать эти оценки нельзя.
''', encoding='utf-8')
    print(f'Published clock-proxy models and results: {OUT}')


if __name__ == '__main__':
    main()
