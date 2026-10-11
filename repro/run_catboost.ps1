<#
Обучение CatBoost по протоколу статьи (рис. 2–4, S1–S4, таблицы 2–6) на мощной машине (i9 + RTX 5080).

Запуск из корня репозитория:
    powershell -ExecutionPolicy Bypass -File repro\run_catboost.ps1               # CPU (рекомендуется)
    powershell -ExecutionPolicy Bypass -File repro\run_catboost.ps1 -Device GPU   # GPU, с откатом на CPU
    powershell -ExecutionPolicy Bypass -File repro\run_catboost.ps1 -Jobs 12 -Threads 2

Число процессов подбирается по физическим ядрам и памяти (каждый процесс пайплайна держит ~1.2 ГБ).
Логи: outputs\logs\catboost_<задача>.log, картинки: outputs\figures\CatBoost_*.png и общие figure_*.png.
#>
param(
    [ValidateSet("CPU", "GPU")] [string]$Device = "CPU",
    [int]$Jobs = 0,
    [int]$Threads = 2,
    [int]$SwTrials = 50,
    [int]$ThreeClassTrials = 20
)
$ErrorActionPreference = "Continue"  # stderr от Python не должен останавливать скрипт; ошибки ловим по $LASTEXITCODE
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

# --- ресурсы машины
$cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
$ramGb = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if ($Jobs -le 0) {
    $byCpu = [math]::Max(1, [math]::Floor($cores / $Threads))
    $byRam = [math]::Max(1, [math]::Floor(($ramGb - 4) / 1.2))
    $Jobs = [math]::Min($byCpu, $byRam)
}
Write-Host "Физических ядер: $cores, RAM: $ramGb ГБ -> процессов: $Jobs, потоков на модель: $Threads, устройство: $Device"

# --- зависимости
& $Python -c "import catboost" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ставлю catboost..."
    & $Python -m pip install "catboost>=1.2"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить catboost" }
}
& $Python -c "import catboost; print('catboost', catboost.__version__)"

# --- окружение для пайплайна
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:MPLBACKEND = "Agg"
$env:SLEEP_N_JOBS = "$Jobs"
$env:SLEEP_CATBOOST_THREADS = "$Threads"
$env:SLEEP_CATBOOST_DEVICE = $Device
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "$Threads"

# --- признаки: если их нет, считаем (нужна папка data\ с датасетом PhysioNet)
$featureCount = (Get-ChildItem "outputs\features\*_count_feature.out" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($featureCount -lt 31) {
    Write-Host "Признаков нет ($featureCount из 31) — запускаю препроцессинг (~3 мин)..."
    & $Python source\preprocessing\preprocessing_runner.py
    if ($LASTEXITCODE -ne 0) { throw "Препроцессинг упал" }
}

New-Item -ItemType Directory -Force "outputs\logs" | Out-Null
$tasks = @(
    @{ Name = "mc_sw"; Trials = $SwTrials },
    @{ Name = "mc_3class"; Trials = $ThreeClassTrials },
    @{ Name = "time_features"; Trials = $SwTrials }
)
foreach ($t in $tasks) {
    $log = "outputs\logs\catboost_$($t.Name).log"
    $start = Get-Date
    Write-Host "`n=== $($t.Name) ($($t.Trials) разбиений) -> $log"
    & $Python repro\02_run_original_analysis.py $t.Name $t.Trials --only CatBoost 2>&1 |
        Where-Object { $_ -notmatch "Completed data split" } | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "$($t.Name) упал, см. $log" }
    Write-Host ("{0} готово за {1:N1} мин" -f $t.Name, ((Get-Date) - $start).TotalMinutes)
}

& $Python repro\02_run_original_analysis.py combine
Write-Host "`nГотово. Картинки: outputs\figures\CatBoost_*.png и общие outputs\figures\figure_*.png"
