# Данные

**Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography**, v1.0.0
(Walch et al., 2019) — [PhysioNet: sleep-accel](https://physionet.org/content/sleep-accel/1.0.0/).
Лицензия: Open Data Commons Attribution License v1.0 (`LICENSE.txt`). При использовании ссылайтесь на статью
Walch O. et al., *SLEEP* 42(12), zsz180 (2019), doi:10.1093/sleep/zsz180, и на PhysioNet.

31 испытуемый, одна ночь PSG у каждого. Время везде — секунды от начала записи PSG.

| папка | содержимое | в репозитории |
|---|---|---|
| `labels/` | `<id>_labeled_sleep.txt`: `время стадия` по 30 с (−1 не размечено, 0 wake, 1–4 N1–N4, 5 REM) | да |
| `heart_rate/` | `<id>_heartrate.txt`: `время,уд/мин`, PPG Apple Watch | да |
| `steps/` | `<id>_steps.txt`: `время,шаги`, бины по 10 мин за неделю до ночи | да |
| `motion/` | `<id>_acceleration.txt`: `время ax ay az` в g, ~50 Гц — **2.2 ГБ, файлы до 399 МБ** | **нет**, см. ниже |
| `circadian_predictions/` | выход модели циркадных часов (`время,x,xc`), посчитан нашим кодом из `steps/` | да |
| `mesa/` | данные MESA (NSRR, по заявке) | нет |

## Как докачать ускорение

GitHub не принимает файлы больше 100 МБ, поэтому `motion/` нужно скачать с PhysioNet:

```bash
python repro/download_data.py
```

Скрипт качает файлы параллельно, сверяет каждый с `SHA256SUMS.txt` и пропускает уже скачанные с верной суммой.
Можно и вручную: архив на странице датасета → папку `motion` положить сюда.

Без `motion/` работают все анализы на готовых признаках (`outputs/features/`, `outputs/repro/ext_features/`):
скрипты `repro/02`–`06` и ноутбук после раздела 2. Ускорение нужно только для препроцессинга
(`source/preprocessing/preprocessing_runner.py`) и `repro/01_explore_dataset.py`.

Подробное описание данных и признаков — [docs/DATASET_RU.md](../docs/DATASET_RU.md).
