"""Build a standalone, pre-rendered notebook from saved CatBoost LOSO results."""
import base64
import contextlib
import gzip
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results/catboost_clock_proxy'
sys.path.insert(0, str(ROOT / 'repro'))


def main():
    manifest = json.loads((RESULTS / 'manifest.json').read_text(encoding='utf-8'))
    with np.load(RESULTS / 'oof_predictions.npz', allow_pickle=False) as saved:
        epoch_labels = saved['y']
        epoch_subjects = saved['subjects']
        pred = {name: saved[name] for name in manifest['feature_sets']}
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, y=epoch_labels, subjects=epoch_subjects, **pred)
    tables = {'participant_metrics': pd.read_csv(RESULTS / 'participant_metrics.csv').to_dict('records'),
              'pooled_metrics': pd.read_csv(RESULTS / 'results.csv').to_dict('records'), 'manifest': manifest}
    tables_b64 = base64.b64encode(gzip.compress(json.dumps(tables, ensure_ascii=False).encode())).decode()
    arrays_b64 = base64.b64encode(array_buffer.getvalue()).decode()
    cells = []
    def md(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})
    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.splitlines(keepends=True), 'execution_count': None, 'outputs': []})
    md('''# CatBoost · clock proxy: три дополнительных признака

Готовые результаты двух вариантов CatBoost с clock proxy: исходные признаки и исходные + три быстрых.

**Обучать ничего не нужно.** Таблицы и предсказания уже включены в ноутбук. Он работает самостоятельно, в том числе в Colab; достаточно NumPy, pandas, matplotlib и scikit-learn. Рисунки уже показаны ниже. Для редактирования подписи меняются в ячейке `TEXTS` или непосредственно в ячейках построения.

## Как получены результаты

- 31 участник, 25 481 эпоха по 30 секунд; сон / бодрствование.
- 31 разбиение leave-one-subject-out: тестовый участник полностью исключён из обучения.
- 100 деревьев, глубина 4, learning_rate=0,1, l2_leaf_reg=3, random_seed=0, веса классов balanced.
- Порог вероятности сна **0,5**; сглаживания предсказаний нет.
- Это текущая конфигурация `FAST=True`, но проверка выполнена на всех участниках, без сокращения выборки. Поиска гиперпараметров не было.
- Финальные веса на всех участниках сохранялись отдельно и не использовались для получения этих оценок.
''')
    code('''# Подписи графиков и названия метрик можно редактировать здесь.
TEXTS = {
    "base": "Исходные",
    "fast": "+3 быстрых",
    "auc": "ROC AUC",
    "kappa": "Cohen κ",
    "accuracy": "Accuracy",
    "sleep_correct": "Сон распознан",
    "wake_correct": "Бодрствование распознано",
    "means_title": "CatBoost · clock proxy · средние по участникам · LOSO, порог 0,5",
    "delta_title": "Изменение κ у каждого участника: +3 быстрых − исходные",
    "roc_title": "ROC · объединённые тестовые эпохи",
    "pr_title": "Precision–Recall для бодрствования",
    "confusion_title": "Ошибки при фиксированном пороге 0,5",
}
COLORS = {"base": "#64748b", "fast": "#0f766e"}
''')
    code('''import base64, gzip, io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path
EXPORT_DIR = Path("catboost_analysis_exports")
EXPORT_DIR.mkdir(exist_ok=True)
TABLE_PAYLOAD = "''' + tables_b64 + '''"
ARRAY_PAYLOAD = "''' + arrays_b64 + '''"
snapshot = json.loads(gzip.decompress(base64.b64decode(TABLE_PAYLOAD)))
participant = pd.DataFrame(snapshot["participant_metrics"])
pooled = pd.DataFrame(snapshot["pooled_metrics"])
with np.load(io.BytesIO(base64.b64decode(ARRAY_PAYLOAD)), allow_pickle=False) as stored:
    arrays = {k: stored[k] for k in stored.files}
means = participant.groupby("feature_set")[["auc", "kappa", "accuracy", "sleep_correct", "wake_correct"]].mean()
def feature_set(base, extra=False):
    return f"Motion, HR, {base}" + (" + быстрые" if extra else "")
def name_for(name):
    return "Clock proxy" + " · " + TEXTS["fast" if "быстрые" in name else "base"]
''')
    md('''## 1. Основные результаты: каждый участник имеет равный вес

Для каждого человека метрики рассчитаны по его отложенным предсказаниям, затем усреднены по 31 участнику.
''')
    code('''display(means.rename(columns={k: TEXTS[k] for k in means.columns}).round(4))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["auc", "kappa"]):
    for i, base in enumerate(["Clock"]):
        for extra, offset in [(False, -.18), (True, .18)]:
            key = "fast" if extra else "base"
            value = means.loc[feature_set(base, extra), metric]
            ax.bar(i + offset, value, width=.36, color=COLORS[key], label=TEXTS[key] if i == 0 else None)
            ax.text(i + offset, value + .004, f"{value:.3f}", ha="center")
    ax.set_xticks([0], ["Clock proxy"])
    ax.set_title(TEXTS[metric])
    ax.set_ylim(0, means[metric].max() * 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle(TEXTS["means_title"])
fig.tight_layout()
fig.savefig(EXPORT_DIR / "mean_metrics.png", dpi=180)
plt.show()
''')
    md('''**Результат:** при clock proxy дополнительные признаки улучшили средние AUC с 0,912 до 0,922 и κ с 0,377 до 0,409.
Clock proxy описывает циркадный ритм средствами проекта по истории активности. Это исходный входной признак, его знак не меняется.
''')
    md('''## 2. За счёт чего меняется качество

Сравним распознавание обоих классов. При balanced-весах и пороге 0,5 модель часто предсказывает бодрствование; значения κ нельзя напрямую сравнивать с MLP, обученным по другому протоколу.
''')
    code('''tradeoff = []
for base in ["Clock"]:
    a, b = means.loc[feature_set(base)], means.loc[feature_set(base, True)]
    tradeoff.append({"Время": base, **{TEXTS[m] + " · изменение, п.п.": 100 * (b[m] - a[m])
                    for m in ["accuracy", "sleep_correct", "wake_correct"]}})
display(pd.DataFrame(tradeoff).set_index("Время").round(2))
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, metric in zip(axes, ["accuracy", "sleep_correct", "wake_correct"]):
    for i, base in enumerate(["Clock"]):
        for extra, offset in [(False, -.18), (True, .18)]:
            key = "fast" if extra else "base"
            value = 100 * means.loc[feature_set(base, extra), metric]
            ax.bar(i + offset, value, width=.36, color=COLORS[key], label=TEXTS[key] if i == 0 else None)
            ax.text(i + offset, value + .4, f"{value:.1f}", ha="center", fontsize=9)
    ax.set_xticks([0], ["Clock proxy"])
    ax.set_title(TEXTS[metric])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=.2)
axes[0].set_ylabel("% · среднее по участникам")
axes[-1].legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(EXPORT_DIR / "class_tradeoff.png", dpi=180)
plt.show()
''')
    md('''При добавлении признаков основное изменение — **лучшее распознавание сна**. Среднее распознавание бодрствования почти сохраняется: оно немного снижается: 81,1% → 80,7%. Поэтому фраза «новые признаки улучшают распознавание бодрствования» не описывает этот результат при пороге 0,5.
''')
    md('''## 3. Улучшение не одинаково для всех

Ниже — парные изменения κ у каждого человека. Положительное значение означает улучшение при добавлении признаков. Расчёт использует одних и тех же участников в обоих вариантах.
''')
    code('''deltas = {}
analysis_rows = []
rng = np.random.default_rng(42)
for base in ["Clock"]:
    for metric in ["auc", "kappa"]:
        paired = participant[participant.feature_set.isin([feature_set(base), feature_set(base, True)])].pivot(index="subject", columns="feature_set", values=metric)
        delta = paired[feature_set(base, True)] - paired[feature_set(base)]
        deltas[(base, metric)] = delta
        draws = rng.choice(delta.to_numpy(), size=(10000, len(delta)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [.025, .975])
        analysis_rows.append({"Время": base, "Метрика": TEXTS[metric], "Среднее изменение": delta.mean(),
                              "Улучшилось участников": int((delta > 0).sum()), "Всего": len(delta),
                              "95% интервал · низ": lo, "95% интервал · верх": hi})
paired_analysis = pd.DataFrame(analysis_rows)
display(paired_analysis.round(4))
fig, axes = plt.subplots(1, 1, figsize=(7, 9))
for ax, base in zip(np.atleast_1d(axes), ["Clock"]):
    delta = deltas[(base, "kappa")].sort_values()
    colors = [COLORS["fast"] if v >= 0 else "#b45309" for v in delta]
    ax.barh(range(len(delta)), delta.to_numpy(), color=colors)
    ax.set_yticks(range(len(delta)), delta.index, fontsize=8)
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_title(base)
    ax.set_xlabel("Δ " + TEXTS["kappa"])
    ax.grid(axis="x", alpha=.15)
fig.suptitle(TEXTS["delta_title"])
fig.tight_layout()
fig.savefig(EXPORT_DIR / "participant_deltas.png", dpi=180)
plt.show()
''')
    md('''Интервалы получены описательным bootstrap по участникам: 10 000 повторных выборок парных различий. Они не учитывают весь предыдущий поиск признаков и моделей на этой выборке. Это оценка разброса наблюдаемого эффекта, а не подтверждение переноса на новую популяцию.
''')
    md('''## 4. ROC и Precision–Recall

Теперь объединяем все отложенные эпохи. Здесь каждый человек получает вес в соответствии с длиной записи. Для PR положительный класс — **бодрствование**, чтобы показать качество распознавания редкого класса.
''')
    code('''y = arrays["y"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
curve_rows = []
for name in snapshot["manifest"]["feature_sets"]:
    p = arrays[name]
    color = COLORS["fast" if "быстрые" in name else "base"]
    style = "-"
    auc = roc_auc_score(y, p)
    ap = average_precision_score(1 - y, 1 - p)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(1 - y, 1 - p)
    axes[0].plot(fpr, tpr, color=color, linestyle=style, label=f"{name_for(name)} · {auc:.3f}")
    axes[1].plot(recall, precision, color=color, linestyle=style, label=f"{name_for(name)} · AP {ap:.3f}")
    curve_rows.append({"Набор": name, "AUC · все эпохи": auc, "AP бодрствования": ap})
axes[0].plot([0, 1], [0, 1], ":", color="#999999")
axes[1].axhline((1 - y).mean(), linestyle=":", color="#999999", label="Доля бодрствования")
axes[0].set(title=TEXTS["roc_title"], xlabel="Бодрствование принято за сон", ylabel="Сон распознан")
axes[1].set(title=TEXTS["pr_title"], xlabel="Полнота для бодрствования", ylabel="Точность для бодрствования")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
fig.tight_layout()
fig.savefig(EXPORT_DIR / "roc_pr.png", dpi=180)
plt.show()
display(pd.DataFrame(curve_rows).round(4))
''')
    md('''**Почему AUC здесь отличается от первой таблицы?** Средний AUC по людям оценивает разделение классов внутри каждой записи. Объединённый AUC также сравнивает вероятности эпох разных участников. Это разные расчёты; нельзя смешивать их в одной строке сравнения. κ также нелинейна: κ общей матрицы ошибок отличается от среднего κ отдельных людей.
''')
    md('''## 5. Матрицы ошибок при пороге 0,5

Строки — истинный класс, столбцы — предсказанный. Цвет и проценты нормированы по строке; абсолютное число эпох показано в каждой клетке.
''')
    code('''fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes.flat, snapshot["manifest"]["feature_sets"]):
    matrix = confusion_matrix(y, arrays[name] >= .5, labels=[0, 1])
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]} эпох\\n{fractions[row, col]:.1%}", ha="center", va="center",
                    color="white" if fractions[row, col] > .5 else "black")
    ax.set_xticks([0, 1], ["Бодрствование", "Сон"])
    ax.set_yticks([0, 1], ["Бодрствование", "Сон"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинный класс")
    ax.set_title(name_for(name), fontsize=10)
fig.suptitle(TEXTS["confusion_title"])
fig.tight_layout()
fig.savefig(EXPORT_DIR / "confusion_matrices.png", dpi=180)
plt.show()
''')
    md('''## 6. Описательные результаты при 90% распознанного сна

В исходном скрипте порог для этой таблицы выбирается **по истинным тестовым меткам**, чтобы получить заданную чувствительность. Это анализ возможного рабочего режима, как в таблицах статьи. Он не показывает результат заранее выбранного порога и не должен подменять оценку при 0,5.
''')
    code('''display(pooled[["feature_set", "accuracy@90", "wake_correct@90", "kappa@90", "f1_wake@90"]].round(4))
''')
    md('''## Вывод

Три дополнительных признака улучшили CatBoost с clock proxy. κ улучшилась у 26 из 31 участника. При пороге 0,5 улучшение главным образом связано с распознаванием сна; эффект различается между участниками.

Ограничения:

- Использованы фиксированные настройки на 100 деревьев; результаты не характеризуют все возможные настройки CatBoost.
- Выборка уже использовалась для исследования других моделей и признаков. Нужна независимая внешняя проверка.
- Медиана пульса и центрированные окна используют будущие данные: анализ предназначен для готовой записи.
- С прежним MLP числа напрямую не сопоставимы: отличаются разбиения, веса классов и настройки. Для сравнения моделей нужен общий протокол.
- Этот анализ оценивает три добавки вместе. Вклад каждого признака в прибавку качества отдельно не установлен.

## Веса и воспроизведение

Финальные модели и JSON с порядком входов лежат в [репозитории](https://github.com/AlisaPopova25/ai_school_october/tree/main/results/catboost_clock_proxy). Этот ноутбук не использует финальные веса для оценки и не запускает обучение.

Для повторного обучения только с Clock: `python repro/10_fast_features.py --models CatBoost --base Clock`.
Пересборка этого ноутбука использует опубликованные отложенные предсказания: `python repro/build_catboost_clock_notebook.py`.
''')
    code('''# Сохранить все таблицы анализа рядом с графиками.
participant.to_csv(EXPORT_DIR / "participant_metrics.csv", index=False)
means.to_csv(EXPORT_DIR / "participant_mean_metrics.csv")
paired_analysis.to_csv(EXPORT_DIR / "paired_changes.csv", index=False)
pooled.to_csv(EXPORT_DIR / "pooled_metrics.csv", index=False)
print("Графики и таблицы сохранены в:", EXPORT_DIR)
''')
    # Render requested figures/tables into notebook outputs without an IPython dependency.
    state = {'__name__': '__notebook__'}
    count = 0
    for item in cells:
        if item['cell_type'] != 'code':
            continue
        count += 1
        item['execution_count'] = count
        outputs = item['outputs']
        def display(value):
            if isinstance(value, pd.DataFrame):
                outputs.append({'output_type': 'display_data', 'data': {'text/html': value.to_html(), 'text/plain': value.to_string()}, 'metadata': {}})
            else:
                outputs.append({'output_type': 'display_data', 'data': {'text/plain': str(value)}, 'metadata': {}})
        def show(*args, **kwargs):
            for number in plt.get_fignums():
                buffer = io.BytesIO()
                plt.figure(number).savefig(buffer, format='png', dpi=140, bbox_inches='tight')
                outputs.append({'output_type': 'display_data', 'data': {'image/png': base64.b64encode(buffer.getvalue()).decode()}, 'metadata': {}})
            plt.close('all')
        state['display'] = display
        plt.show = show
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            # Keep notebook export paths portable; generated files go to the snapshot locally.
            source = ''.join(item['source'])
            if 'EXPORT_DIR = Path(' in source:
                source = source.replace('Path("catboost_analysis_exports")', repr(str(RESULTS / 'analysis_exports')))
                source = source.replace('EXPORT_DIR = ' + repr(str(RESULTS / 'analysis_exports')), 'EXPORT_DIR = Path(' + repr(str(RESULTS / 'analysis_exports')) + ')')
            exec(compile(source, '<notebook-cell>', 'exec'), state)
        if stdout.getvalue():
            outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stdout.getvalue()})
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
    path = RESULTS / 'catboost_clock_analysis.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(state['paired_analysis'].to_string(index=False))
    print(path)


if __name__ == '__main__':
    main()
