"""Обучение CatBoost по протоколу статьи Walch et al. (2019): рисунки 2–4, S1–S4 и таблицы 2–6.

Как запустить
-------------
VS Code: Run and Debug (Ctrl+Shift+D) → «6. Train CatBoost (CPU)» → F5.
Терминал, из корня репозитория:

    python repro/train_catboost.py                 # полный протокол на CPU (рекомендуется)
    python repro/train_catboost.py --dry-run       # только показать план, ничего не обучать
    python repro/train_catboost.py --device GPU    # GPU; если не заведётся — CatBoost сам перейдёт на CPU
    python repro/train_catboost.py --jobs 4 --sw-trials 20 --three-class-trials 10   # облегчённо (ноутбук)
    python repro/train_catboost.py --tasks mc_sw   # только одна задача

Что делает
----------
1. Определяет ядра и свободную память и выбирает число параллельных процессов
   (каждый процесс пайплайна держит ~1.2 ГБ памяти).
2. Ставит catboost, если его нет.
3. Проверяет, что готовы признаки (outputs/features); если нет — считает их из data/ (нужен датасет).
4. По очереди запускает задачи оригинального пайплайна только для CatBoost:
     mc_sw          — сон/бодрствование, Monte Carlo (рис. 2–3, табл. 2–5)
     mc_3class      — wake/NREM/REM (рис. 4, S1–S3, табл. 6)
     time_features  — clock / cosine / time (рис. S4)
   Каждая задача — отдельный процесс `python repro/02_run_original_analysis.py <задача> <разбиения> --only CatBoost`:
   ограничения потоков (OpenBLAS и др.) нужно задать ДО старта Python, поэтому задачи запускаются как дочерние процессы.
5. Собирает общие картинки со всеми моделями (`02_run_original_analysis.py combine`)
   и все таблицы в читаемый outputs/RESULTS.md (`08_results_markdown.py`).

Настройки (параметры модели, перебор, число разбиений) — ОДИН файл:
    source/analysis/classification/catboost_config.py
Там переключатель FAST: True — быстро (минуты, проверить запуск), False — полный протокол статьи (часы).

Результаты: таблицы и ход работы — outputs/logs/catboost_<задача>.log,
картинки — outputs/figures/CatBoost_*.png и общие outputs/figures/figure_*.png.
Сама модель — source/analysis/classification/catboost_classifier.py.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from source.analysis.classification import catboost_config  # noqa: E402  — все настройки CatBoost здесь

CONFIG_FILE = "source/analysis/classification/catboost_config.py"
DRIVER = ROOT / "repro" / "02_run_original_analysis.py"
LOGS = ROOT / "outputs" / "logs"
GB_PER_PROCESS = 1.2  # измерено: один рабочий процесс пайплайна с загруженными признаками

TASKS = {  # задача -> описание (число разбиений берётся из аргументов)
    "mc_sw": "сон/бодрствование, Monte Carlo — рис. 2–3, табл. 2–5",
    "mc_3class": "wake/NREM/REM — рис. 4, S1–S3, табл. 6",
    "time_features": "clock / cosine / time — рис. S4",
}


# --------------------------------------------------------------------------- ресурсы машины
def machine_resources():
    """(физические ядра, логические потоки, свободная RAM в ГБ, общая RAM в ГБ)."""
    logical = os.cpu_count() or 1
    try:
        import psutil  # ставится вместе с jupyter (ipykernel)
        physical = psutil.cpu_count(logical=False) or logical
        memory = psutil.virtual_memory()
        return physical, logical, memory.available / 2 ** 30, memory.total / 2 ** 30
    except ImportError:
        return max(1, logical // 2), logical, None, None


def choose_jobs(threads, physical, free_gb):
    by_cpu = max(1, physical // threads)
    if free_gb is None:
        return by_cpu
    by_ram = max(1, int((free_gb - 2) / GB_PER_PROCESS))  # 2 ГБ запаса системе
    return max(1, min(by_cpu, by_ram))


# --------------------------------------------------------------------------- шаги
def ensure_catboost(dry_run):
    if importlib.util.find_spec("catboost") is not None:
        import catboost
        print(f"catboost {catboost.__version__} уже установлен")
        return
    command = [sys.executable, "-m", "pip", "install", "catboost>=1.2"]
    print("catboost не установлен ->", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def ensure_features(env, dry_run):
    features = list((ROOT / "outputs" / "features").glob("*_count_feature.out"))
    if len(features) >= 31:
        print(f"признаки готовы: {len(features)} испытуемых в outputs/features")
        return
    motion = list((ROOT / "data" / "motion").glob("*_acceleration.txt"))
    if not motion:
        sys.exit("Нет ни признаков (outputs/features), ни датасета (data/motion).\n"
                 "Скачайте датасет: python repro/download_data.py — и запустите снова.")
    command = [sys.executable, str(ROOT / "source" / "preprocessing" / "preprocessing_runner.py")]
    print(f"признаков нет ({len(features)} из 31) -> препроцессинг (~3 мин):", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT, env=env)


def run_logged(command, env, log_path):
    """Запускает команду, печатает её вывод (без построчного шума) и пишет в лог. Возвращает код выхода."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace") as process:
        try:
            for line in process.stdout:
                if "Completed data split" in line:  # по строке на каждое разбиение — только в лог
                    log.write(line)
                    continue
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            raise
    return process.returncode


def make_env(jobs, threads, device):
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "repro")]),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore",
        "MPLBACKEND": "Agg",
        "SLEEP_N_JOBS": str(jobs),               # сколько разбиений обучается параллельно
        "SLEEP_CATBOOST_THREADS": str(threads),  # потоков CatBoost на одну модель
        "SLEEP_CATBOOST_DEVICE": device,         # CPU или GPU
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": "1",
    })
    return env


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU",
                        help="CPU (по умолчанию) или GPU; при сбое GPU CatBoost сам перейдёт на CPU")
    parser.add_argument("--jobs", type=int, default=0, help="параллельных процессов (0 = подобрать автоматически)")
    parser.add_argument("--threads", type=int, default=2, help="потоков на одну модель CatBoost")
    parser.add_argument("--sw-trials", type=int, default=catboost_config.SW_TRIALS,
                        help=f"разбиений для сна/бодрствования (по умолчанию из {CONFIG_FILE}; в статье 50)")
    parser.add_argument("--three-class-trials", type=int, default=catboost_config.THREE_CLASS_TRIALS,
                        help=f"разбиений для трёх классов (по умолчанию из {CONFIG_FILE}; в статье 20)")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS), help="какие задачи запускать")
    parser.add_argument("--dry-run", action="store_true", help="показать план и выйти, ничего не обучая")
    args = parser.parse_args()

    mode = "БЫСТРЫЙ (FAST = True)" if catboost_config.FAST else "ПОЛНЫЙ протокол статьи (FAST = False)"
    print(f"Режим: {mode}. Настройки: {CONFIG_FILE}")
    print(f"  модель: {catboost_config.PARAMS}")
    print(f"  перебор: {catboost_config.GRID}")
    physical, logical, free_gb, total_gb = machine_resources()
    jobs = args.jobs or choose_jobs(args.threads, physical, free_gb)
    memory = f"свободно {free_gb:.1f} из {total_gb:.1f} ГБ" if free_gb is not None else "объём неизвестен (нет psutil)"
    print(f"Ядер: {physical} физических / {logical} логических; память: {memory}")
    print(f"-> процессов: {jobs}, потоков на модель: {args.threads}, устройство: {args.device}")
    if free_gb is not None and jobs * GB_PER_PROCESS > free_gb:
        print(f"Внимание: {jobs} процессов x {GB_PER_PROCESS} ГБ больше свободной памяти — уменьшите --jobs")

    env = make_env(jobs, args.threads, args.device)
    ensure_catboost(args.dry_run)
    ensure_features(env, args.dry_run)

    trials = {"mc_sw": args.sw_trials, "mc_3class": args.three_class_trials, "time_features": args.sw_trials}
    plan = [(task, trials[task]) for task in TASKS if task in args.tasks]
    print("\nПлан:")
    for task, n in plan:
        print(f"  {task:<14} {n:>3} разбиений  — {TASKS[task]}")
    if args.dry_run:
        print("\n--dry-run: обучение не запускалось. Команды:")
        for task, n in plan:
            print("  ", " ".join([Path(sys.executable).name, str(DRIVER.relative_to(ROOT)), task, str(n), "--only", "CatBoost"]))
        return

    started = time.time()
    for task, n in plan:
        log_path = LOGS / f"catboost_{task}.log"
        print(f"\n=== {task}: {n} разбиений, лог -> {log_path.relative_to(ROOT)}")
        t0 = time.time()
        code = run_logged([sys.executable, str(DRIVER), task, str(n), "--only", "CatBoost"], env, log_path)
        if code != 0:
            sys.exit(f"{task} завершился с ошибкой (код {code}), подробности в {log_path}")
        print(f"=== {task} готово за {(time.time() - t0) / 60:.1f} мин")

    print("\nСобираю общие картинки со всеми моделями...")
    subprocess.run([sys.executable, str(DRIVER), "combine"], cwd=ROOT, env=env, check=True)
    print("Собираю все таблицы в читаемый outputs/RESULTS.md...")
    subprocess.run([sys.executable, str(ROOT / "repro" / "08_results_markdown.py")], cwd=ROOT, env=env, check=True)
    print(f"\nГотово за {(time.time() - started) / 60:.1f} мин. Таблицы — outputs/RESULTS.md (Markdown) "
          f"и outputs/logs/catboost_*.log (LaTeX), картинки — outputs/figures/CatBoost_*.png и figure_*.png")


if __name__ == "__main__":
    main()
