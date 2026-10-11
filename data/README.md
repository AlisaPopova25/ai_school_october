# Данные

Датасет в репозитории не хранится (2.3 ГБ, отдельные файлы до 399 МБ). Его нужно скачать по ссылке.

**Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography**, v1.0.0
(Walch et al., 2019): **https://physionet.org/content/sleep-accel/1.0.0/**

Лицензия — Open Data Commons Attribution v1.0. При использовании ссылайтесь на статью Walch O. et al.,
*SLEEP* 42(12), zsz180 (2019), doi:10.1093/sleep/zsz180, и на PhysioNet.

## Как скачать

Скриптом, из корня репозитория:

```bash
python repro/download_data.py
```

Он качает все 125 файлов параллельно, сверяет каждый с официальными SHA256 и раскладывает по папкам ниже.
Уже скачанные файлы с верной суммой пропускаются, поэтому прерванную загрузку можно просто запустить заново.

Вручную: на странице датасета скачайте ZIP (~577 МБ), распакуйте и положите в эту папку `heart_rate/`, `labels/`,
`motion/`, `steps/`, `LICENSE.txt`, `SHA256SUMS.txt`. Внутри архива бывает вложенная папка с тем же именем —
брать нужно её содержимое.

## Что где будет

| папка | содержимое |
|---|---|
| `labels/` | `<id>_labeled_sleep.txt`: `время стадия` по 30 с (−1 не размечено, 0 wake, 1–4 N1–N4, 5 REM) |
| `heart_rate/` | `<id>_heartrate.txt`: `время,уд/мин`, PPG Apple Watch |
| `steps/` | `<id>_steps.txt`: `время,шаги`, бины по 10 мин за неделю до ночи |
| `motion/` | `<id>_acceleration.txt`: `время ax ay az` в g, ~50 Гц (2.2 ГБ) |
| `circadian_predictions/` | создаётся препроцессингом: выход модели циркадных часов по шагам |
| `mesa/` | данные MESA (NSRR, по заявке) — не используются |

31 испытуемый, одна ночь PSG у каждого; время везде — секунды от начала записи PSG.

## Нужен ли датасет

Нет, если запускаете только анализ: готовые признаки лежат в `outputs/features/` и `outputs/repro/ext_features/`,
на них работают `repro/02`–`06` и ноутбук начиная с раздела 7. Датасет нужен для препроцессинга
(`source/preprocessing/preprocessing_runner.py`), `repro/01_explore_dataset.py` и первых разделов ноутбука.

Подробное описание данных и признаков — [docs/DATASET_RU.md](../docs/DATASET_RU.md).
