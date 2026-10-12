# Плакат A0 (команда 11)

Плакат собирается скриптом из шаблона `template.pptx`: шапка (команда, авторы, ментор, статья) и две колонки
карточек с текстом, таблицами, формулами и рисунками. Высоты блоков рассчитываются заранее, чтобы ничего не
вылезало за карточки.

```bash
python poster/build_poster.py --team 11        # -> poster/Команда_11_Постер.pptx (если файл открыт — ..._v2.pptx)
python poster/preview.py poster/Команда_11_Постер.pptx poster/preview.png   # приблизительное превью без PowerPoint
python poster/prepare_assets.py .              # пересобрать картинки и формулы в poster/assets/
```

* Тексты карточек — в `build_poster.py` (списки `LEFT`, `RIGHT`, `CONCLUSIONS`).
* Если уже посчитан `python repro/10_fast_features.py`, в карточку 6 автоматически добавится таблица
  «база против база + быстрые» из `outputs/repro/fast_features/results.csv`.
* Если положить картинку SHAP расширенной модели в `poster/assets/shap_fast.png` (две панели рядом), в карточку 6
  добавится её правая половина.
* Формулы — картинки; их LaTeX-исходники для вставки в PowerPoint (Вставка → Уравнение → LaTeX) —
  `assets/formulas_latex.txt`.
* PDF для печати — из PowerPoint: Файл → Экспорт → PDF («Стандартная: публикация и печать»).
