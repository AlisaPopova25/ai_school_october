"""ОДНА команда: обучить модели, построить рисунки и собрать ВСЁ в один Markdown-файл outputs/RESULTS.md.

Запуск из корня репозитория (или VS Code → Run and Debug → «0. RUN ALL»):

    python repro/run_all.py                 # метрики статьи для 4 моделей статьи + CatBoost и RESULTS.md
    python repro/run_all.py --all           # плюс все остальные анализы проекта (см. STEPS ниже)
    python repro/run_all.py --skip-nn       # без нейросети (быстрее; дообучить потом можно без --skip-nn)
    python repro/run_all.py --skip-catboost # без CatBoost
    python repro/run_all.py --steps f1 report      # только выбранные шаги
    python repro/run_all.py --dry-run       # показать план и выйти, ничего не запуская
    python repro/run_all.py --smoke         # проверка, что всё запускается: 6 человек, крошечные модели,
                                            # результаты в outputs/repro/smoke/ (настоящие не трогаются)

Во время работы видно: этап «i/N», для обучения — модель, набор признаков, задача и прогресс-бар по фолдам
с оценкой оставшегося времени. Каждый шаг — отдельный процесс со своим логом outputs/logs/run_all_<шаг>.log;
при ошибке шага скрипт останавливается и говорит, где смотреть. Шаг report выполняется последним всегда.
Предсказания моделей кэшируются (outputs/repro/paper_metrics/oof/): повторный запуск не переобучает то,
что уже посчитано, — чтобы переобучить, удалите эту папку. CatBoost берёт параметры из
source/analysis/classification/catboost_config.py (FAST = True — быстро, False — полный режим).
"""
import argparse
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parent
ROOT = REPRO.parent
sys.path.insert(0, str(REPRO))
from progress import fmt_time  # noqa: E402
from train_catboost import choose_jobs, ensure_catboost, machine_resources, make_env, run_logged  # noqa: E402

MODELS = ["Logistic Regression", "k-Nearest Neighbors", "Random Forest", "Neural Net", "CatBoost"]

# шаг -> (описание, список команд). Команда — аргументы python; лог — outputs/logs/run_all_<шаг>.log
STEPS = {
    "paper_metrics": ("все метрики статьи (табл. 1–6, рис. 2–7) + F1, AP, PPV и важность признаков по Шепли/SHAP: "
                      "4 модели статьи + CatBoost", None),
    "fast_features": ("быстрые признаки команды (пик движения, доля активных отрезков, относительный пульс): "
                      "база против база + быстрые для CatBoost и нейросети", None),
    "original_lr": ("оригинальный код статьи, логистическая регрессия (табл. 2, 6, рис. S4)",
                    [["repro/02_run_original_analysis.py", task, "--only", "Logistic Regression"]
                     for task in ("mc_sw", "mc_3class", "time_features")]),
    "importance": ("важность признаков: Шепли, веса, permutation, SHAP + картинки с пояснениями",
                   [["repro/03_feature_importance.py"], ["repro/make_shapley_guide.py"]]),
    "improve": ("расширенные признаки, LightGBM, HMM", [["repro/04_improve_model.py"]]),
    "logreg": ("логистическая регрессия: коэффициенты, интервалы, калибровка", [["repro/05_logreg_deep_dive.py"]]),
    "time_of_night": ("качество по часам ночи", [["repro/06_quality_by_time.py"]]),
    "f1": ("F1 против accuracy, F1 по стадиям и людям", [["repro/07_f1_metrics.py"]]),
    "dataset": ("обзор датасета (нужна папка data/ с датасетом)", [["repro/01_explore_dataset.py"]]),
    "report": ("собрать всё в outputs/RESULTS.md", [["repro/08_results_markdown.py"]]),
}
DEFAULT_STEPS = ["paper_metrics", "fast_features", "report"]
ALL_STEPS = ["paper_metrics", "fast_features", "original_lr", "importance", "improve", "logreg", "time_of_night", "f1", "dataset",
             "report"]


def main():
    parser = argparse.ArgumentParser(description="Обучить, нарисовать и собрать всё в outputs/RESULTS.md")
    parser.add_argument("--all", action="store_true", help="все шаги проекта, а не только метрики статьи")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), help="только эти шаги (report добавится сам)")
    parser.add_argument("--skip-nn", action="store_true", help="не обучать нейросеть")
    parser.add_argument("--skip-catboost", action="store_true", help="не обучать CatBoost")
    parser.add_argument("--jobs", type=int, default=0, help="параллельных процессов (0 = по ядрам и памяти)")
    parser.add_argument("--dry-run", action="store_true", help="только показать план")
    parser.add_argument("--smoke", action="store_true",
                        help="быстрая проверка: 6 человек, крошечные модели, результаты в outputs/repro/smoke/")
    args = parser.parse_args()

    steps = args.steps or (ALL_STEPS if args.all else DEFAULT_STEPS)
    if args.smoke:
        steps = ["paper_metrics", "fast_features"]  # у этих шагов есть проверочный режим; отчёт — в папку smoke
    steps = [s for s in ALL_STEPS if s in steps and s != "report"] + ["report"]  # порядок фиксирован, report — в конце
    if "dataset" in steps and not any((ROOT / "data" / "motion").glob("*_acceleration.txt")):
        print("шаг dataset пропущен: нет датасета в data/ (скачать: python repro/download_data.py)")
        steps.remove("dataset")

    physical, logical, free_gb, total_gb = machine_resources()
    jobs = args.jobs or choose_jobs(1, physical, free_gb)
    env = make_env(jobs, 1, "CPU")
    env["SLEEP_INCLUDE_CATBOOST"] = "0"  # оригинальный пайплайн (шаг original_lr) — только логистическая регрессия
    memory = f", свободно {free_gb:.1f} ГБ" if free_gb is not None else ""
    print(f"Ядер: {physical}{memory} -> параллельных процессов: {jobs}")

    models = [m for m in MODELS if not (args.skip_nn and m == "Neural Net")
              and not (args.skip_catboost and m == "CatBoost")]
    if ("paper_metrics" in steps or "fast_features" in steps) and "CatBoost" in models:
        ensure_catboost(args.dry_run)  # ставит catboost, если его нет
    plan = []
    for step in steps:
        commands = STEPS[step][1]
        if step == "paper_metrics":
            commands = [["repro/09_paper_metrics.py", "--no-report", "--models", *models]
                        + (["--smoke"] if args.smoke else [])]
        if step == "fast_features":
            fast_models = [m for m in models if m in ("CatBoost", "Neural Net")]
            commands = [["repro/10_fast_features.py", "--models", *fast_models] + (["--smoke"] if args.smoke else [])]
        if step == "report" and args.smoke:
            smoke_dir = str(ROOT / "outputs" / "repro" / "smoke")
            commands = [["repro/08_results_markdown.py", "--analysis-dir", smoke_dir,
                         "--out", str(Path(smoke_dir) / "RESULTS.md")]]
        plan.append((step, commands))

    print("\nПлан:")
    for i, (step, commands) in enumerate(plan, 1):
        print(f"  {i}. {step:<14} — {STEPS[step][0]}")
        for c in commands:
            print("      python " + " ".join(f'"{a}"' if " " in a else a for a in c))
    if args.dry_run:
        print("\n--dry-run: ничего не запускалось")
        return

    started = time.time()
    for i, (step, commands) in enumerate(plan, 1):
        for k, command in enumerate(commands, 1):
            suffix = f" {command[1]}" if step == "original_lr" else (f" ({k}/{len(commands)})" if len(commands) > 1 else "")
            name = f"run_all_{step}_{k}.log" if len(commands) > 1 else f"run_all_{step}.log"
            log = ROOT / "outputs" / "logs" / (f"lr_{command[1]}.log" if step == "original_lr" else name)
            print(f"\n=== этап {i}/{len(plan)}: {step}{suffix} — {STEPS[step][0]} "
                  f"(прошло {fmt_time(time.time() - started)}, лог: {log.relative_to(ROOT)})", flush=True)
            t0 = time.time()
            code = run_logged([sys.executable, str(ROOT / command[0]), *command[1:]], env, log)
            if code != 0:
                sys.exit(f"Этап {i}/{len(plan)} ({step}) завершился с ошибкой (код {code}), подробности: {log}")
            print(f"=== этап {i}/{len(plan)}: {step}{suffix} готово за {fmt_time(time.time() - t0)}", flush=True)

    target = "outputs/repro/smoke/RESULTS.md" if args.smoke else "outputs/RESULTS.md"
    print(f"\nГотово за {fmt_time(time.time() - started)}. Всё собрано в {target} "
          f"(в VS Code: открыть файл и Ctrl+Shift+V для просмотра)")


if __name__ == "__main__":
    main()
