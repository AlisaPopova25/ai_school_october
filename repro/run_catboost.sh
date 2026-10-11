#!/usr/bin/env bash
# Обучение CatBoost по протоколу статьи (рис. 2–4, S1–S4, таблицы 2–6) — Linux / ВМ.
#   bash repro/run_catboost.sh            # CPU (рекомендуется)
#   DEVICE=GPU bash repro/run_catboost.sh # GPU, с откатом на CPU
#   JOBS=12 THREADS=2 bash repro/run_catboost.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

THREADS=${THREADS:-2}
DEVICE=${DEVICE:-CPU}
CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l || nproc)
RAM_GB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
if [ -z "${JOBS:-}" ]; then
  BY_CPU=$(( CORES / THREADS )); BY_RAM=$(( (RAM_GB - 4) * 10 / 12 ))
  JOBS=$(( BY_CPU < BY_RAM ? BY_CPU : BY_RAM )); [ "$JOBS" -lt 1 ] && JOBS=1
fi
echo "Физических ядер: $CORES, RAM: ${RAM_GB} ГБ -> процессов: $JOBS, потоков на модель: $THREADS, устройство: $DEVICE"

$PY -c "import catboost" 2>/dev/null || $PY -m pip install "catboost>=1.2"
$PY -c "import catboost; print('catboost', catboost.__version__)"

export PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export SLEEP_N_JOBS=$JOBS SLEEP_CATBOOST_THREADS=$THREADS SLEEP_CATBOOST_DEVICE=$DEVICE
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=$THREADS

if [ "$(ls outputs/features/*_count_feature.out 2>/dev/null | wc -l)" -lt 31 ]; then
  echo "Признаков нет — запускаю препроцессинг (~3 мин)..."
  $PY source/preprocessing/preprocessing_runner.py
fi

mkdir -p outputs/logs
for task in "mc_sw 50" "mc_3class 20" "time_features 50"; do
  set -- $task
  echo; echo "=== $1 ($2 разбиений) -> outputs/logs/catboost_$1.log"
  start=$(date +%s)
  $PY repro/02_run_original_analysis.py "$1" "$2" --only CatBoost 2>&1 | grep -v "Completed data split" | tee "outputs/logs/catboost_$1.log"
  echo "$1 готово за $(( ($(date +%s) - start) / 60 )) мин"
done

$PY repro/02_run_original_analysis.py combine
echo; echo "Готово. Картинки: outputs/figures/CatBoost_*.png и общие outputs/figures/figure_*.png"
