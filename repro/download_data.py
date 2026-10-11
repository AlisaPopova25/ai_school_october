"""Параллельная загрузка датасета sleep-accel (PhysioNet) в папку data/ с проверкой SHA256.

Качает только недостающие или повреждённые файлы: файлы с верной контрольной суммой пропускаются.
В git-репозитории проекта лежит всё, кроме data/motion (2.2 ГБ), поэтому на практике скрипт
докачивает только сырое ускорение.

Запуск из корня репозитория:  python repro/download_data.py
"""
import hashlib
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/sleep-accel/1.0.0/"
ROOT = Path(__file__).resolve().parent.parent / "data"
FOLDERS = ["labels", "heart_rate", "steps", "motion"]


def list_files(folder):
    html = urllib.request.urlopen(BASE + folder + "/").read().decode()
    return [h for h in re.findall(r'href="([^"/]+)"', html) if h.endswith(".txt")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel, expected):
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and expected and sha256(dst) == expected:
        return rel, "cached"
    for attempt in range(5):
        try:
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(BASE + rel, tmp)
            tmp.replace(dst)
            if expected and sha256(dst) != expected:
                raise IOError("checksum mismatch")
            return rel, "ok"
        except Exception as e:  # retry on network errors
            err = e
    return rel, f"FAILED: {err}"


def main():
    sums_text = urllib.request.urlopen(BASE + "SHA256SUMS.txt").read().decode()
    sums = {name: h for h, name in (line.split(maxsplit=1) for line in sums_text.splitlines() if line.strip())}
    files = [f"{folder}/{name}" for folder in FOLDERS for name in list_files(folder)]
    files += ["LICENSE.txt", "SHA256SUMS.txt"]
    print(f"{len(files)} files to fetch", flush=True)
    with ThreadPoolExecutor(max_workers=24) as pool:
        for rel, status in pool.map(lambda r: fetch(r, sums.get(r)), files):
            print(f"{status:8s} {rel}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
