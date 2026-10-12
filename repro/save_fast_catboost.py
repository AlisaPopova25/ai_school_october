"""Save final CatBoost weights and an explicit LOSO report after 10_fast_features.py.

Run: python repro/10_fast_features.py --models CatBoost
     python repro/save_fast_catboost.py
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import catboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'repro'))
sys.path.insert(0, str(ROOT))
import sleeplib as sl
import fast_features as ff
from source.analysis.classification import catboost_config


def main():
    spec = importlib.util.spec_from_file_location('fast_experiment', ROOT / 'repro/10_fast_features.py')
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    data = ff.add_to(sl.load_paper_features())
    out = ROOT / 'outputs/repro/fast_features'
    snapshot = ROOT / 'results/catboost_fast_features'
    snapshot.mkdir(parents=True, exist_ok=True)
    settings = experiment.feature_sets(['Clock', 'Cosine'])
    manifest = {'task': 'sleep/wake', 'participants': sl.SUBJECTS, 'threshold': .5,
                'protocol': '31-fold leave-one-subject-out; no test participant in training',
                'catboost_version': catboost.__version__, 'params': catboost_config.PARAMS,
                'class_weight': 'balanced', 'config_fast': catboost_config.FAST,
                'epochs': sum(len(d['stage']) for d in data.values()), 'feature_sets': settings,
                'epoch_metrics': 'Pooled across all held-out epochs',
                'participant_metrics': 'Unweighted mean across participants',
                'sensitivity_90_metrics': 'Descriptive operating point selected using test labels, not a deployed threshold',
                'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                    ['repro/10_fast_features.py', 'repro/fast_features.py',
                     'additional_sleep_features/additional_sleep_features.py',
                     'source/analysis/classification/catboost_config.py']}, 'models': {}}
    rows = []
    for name, features in settings.items():
        file = out / 'oof' / f'catboost__{name}.npz'
        if not file.exists():
            raise FileNotFoundError(f'Run 10_fast_features.py first: {file}')
        manifest.setdefault('oof_sha256', {})[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file) as predictions:
            for subject in sl.SUBJECTS:
                y, p = sl.sleep_wake(data[subject]['stage']), predictions[subject]
                if len(y) != len(p):
                    raise ValueError(f'Misaligned predictions: {subject}')
                rows.append({'feature_set': name, 'subject': subject, 'epochs': len(y),
                    'auc': roc_auc_score(y, p), 'kappa': cohen_kappa_score(y, p >= .5),
                    'accuracy': accuracy_score(y, p >= .5),
                    'sleep_correct': float(np.mean(p[y == 1] >= .5)),
                    'wake_correct': float(np.mean(p[y == 0] < .5))})
        x, stages, _ = sl.stack(data, sl.SUBJECTS, features)
        model = experiment.paper_metrics().make_model('CatBoost', 'sw').fit(x, sl.sleep_wake(stages))
        stem = 'catboost_' + ('cosine' if 'Cosine' in name else 'clock') + ('_fast' if 'быстрые' in name else '_base')
        path = snapshot / (stem + '.cbm')
        model.model_.save_model(str(path))
        info = {'feature_names': features, 'threshold_sleep': .5, 'class_order': [0, 1],
                'labels': {'0': 'wake', '1': 'sleep'}, 'training_participants': 31,
                'training_epochs': len(x), 'parameters': model.get_params(),
                'note': 'Final model trained on all participants; reported metrics come from separate LOSO models'}
        (snapshot / (stem + '.json')).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest['models'][stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'Saved {path.name}: {len(features)} inputs', flush=True)
    subjects = pd.DataFrame(rows)
    subjects.to_csv(snapshot / 'participant_metrics.csv', index=False)
    means = subjects.groupby('feature_set')[['auc', 'kappa', 'accuracy', 'sleep_correct', 'wake_correct']].mean()
    means.to_csv(snapshot / 'participant_mean_metrics.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ['auc', 'kappa'], ['ROC AUC', 'Cohen κ · порог 0,5']):
        for i, base in enumerate(['Clock', 'Cosine']):
            for extra, offset, color, label in [(False, -.18, '#64748b', 'Исходные'),
                                                (True, .18, '#0f766e', '+3 быстрых')]:
                name = f'Motion, HR, {base}' + (' + быстрые' if extra else '')
                value = float(means.loc[name, metric])
                ax.bar(i + offset, value, width=.36, color=color, label=label if i == 0 else None)
                ax.text(i + offset, value + .004, f'{value:.3f}', ha='center', fontsize=10)
        ax.set_xticks([0, 1], ['Clock', 'Cosine'])
        ax.set_title(title)
        ax.set_ylim(0, float(means[metric].max()) * 1.15)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('CatBoost: исходные признаки и +3 быстрых\nLOSO · средние по 31 участнику · 100 деревьев, глубина 4')
    fig.tight_layout()
    fig.savefig(snapshot / 'comparison_fixed_threshold.png', dpi=180)
    plt.close(fig)
    import shutil
    for name in ['results.csv', 'results.md', 'fig_fast_features.png']:
        shutil.copy2(out / name, snapshot / name)
    (snapshot / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    pooled = pd.read_csv(snapshot / 'results.csv')
    text = '# CatBoost с тремя дополнительными признаками\n\n'
    text += 'Обучены четыре варианта: Motion + HR + Clock/Cosine, с тремя быстрыми признаками и без них. Дополнительные признаки: пик движения за эпоху, доля активных пятисекундных отрезков за минуту, относительный пульс.\n\n'
    text += '## Протокол\n\n31 участник, 25 481 эпоха. Leave-one-subject-out: каждый участник предсказывается моделью, обученной на остальных 30. Всего 124 обучения для оценки и ещё четыре обучения на всех участниках для сохранения весов. Порог P(сон) = 0,5; предсказания не сглаживаются.\n\n'
    text += 'Использована текущая конфигурация репозитория FAST=True: 100 деревьев, глубина 4, learning_rate=0,1, l2_leaf_reg=3, random_seed=0, веса классов balanced. Это полная проверка на 31 участнике с фиксированными параметрами; расширенного поиска гиперпараметров нет.\n\n'
    text += '## Метрики по объединённым тестовым эпохам\n\n| Время | Признаки | AUC | κ при 0,5 |\n|---|---|---:|---:|\n'
    for r in pooled.itertuples(index=False):
        text += f"| {'Cosine' if 'Cosine' in r.feature_set else 'Clock'} | {'+3 быстрых' if 'быстрые' in r.feature_set else 'Исходные'} | {getattr(r, 'auc'):.3f} | {pooled.loc[pooled.feature_set == r.feature_set, 'kappa@0.5'].iloc[0]:.3f} |\n"
    text += '\n## Средние метрики по участникам\n\n| Набор | AUC | κ при 0,5 |\n|---|---:|---:|\n'
    for name, r in means.iterrows():
        text += f'| {name} | {r.auc:.3f} | {r.kappa:.3f} |\n'
    text += '\nДобавление признаков улучшило AUC и κ по объединённым эпохам для обоих вариантов времени. Эти цифры нельзя напрямую сравнивать с прежним MLP: там были пять внешних разбиений и средние по участникам. Для честного сравнения CatBoost и MLP нужны совпадающие модели, разбиения и способ усреднения.\n\n'
    text += 'Показатели при чувствительности ко сну 90% в results.csv и на рисунке — описательный анализ с выбором порога по истинным тестовым меткам, как в таблицах статьи. Это не качество заранее выбранного порога.\n\n'
    text += 'Clock — циркадный признак из репозитория, Cosine — временная косинусоида. Цветовое отрицание clock на прежних SHAP-графиках не применяется к входам модели. Окна и медиана записи используют будущие данные. Выборка уже использовалась в других экспериментах, нужна внешняя проверка.\n\n'
    text += '## Использование весов\n\n```python\nfrom catboost import CatBoostClassifier\nmodel = CatBoostClassifier()\nmodel.load_model("results/catboost_fast_features/catboost_cosine_fast.cbm")\n# X: motion, hr, cosine, peak, active, hr_rel — именно в таком порядке.\np_sleep = model.predict_proba(X)[:, 1]\nsleep = p_sleep >= 0.5\n```\n\n'
    text += 'Порядок признаков и параметры каждой модели записаны в соседнем JSON. Финальные веса не использовались для оценки качества.\n\nПовторение: `python repro/10_fast_features.py --models CatBoost`, затем `python repro/save_fast_catboost.py`.\n'
    (snapshot / 'README.md').write_text(text, encoding='utf-8')
    print(means.to_string(), flush=True)


if __name__ == '__main__':
    main()
