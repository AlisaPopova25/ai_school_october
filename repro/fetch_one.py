"""Докачка одного файла датасета с PhysioNet параллельными Range-запросами + проверка SHA256."""
import hashlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

rel = sys.argv[1]
url = "https://physionet.org/files/sleep-accel/1.0.0/" + rel
root = Path(__file__).resolve().parent.parent / "data"
size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
expected = dict(reversed(l.split(maxsplit=1)) for l in (root / "SHA256SUMS.txt").read_text().splitlines() if l.strip())[rel]
parts = 32
bounds = [(i * size // parts, (i + 1) * size // parts - 1) for i in range(parts)]

def get(b):
    for _ in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={b[0]}-{b[1]}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) == b[1] - b[0] + 1:
                return data
        except Exception:
            pass
    raise IOError(f"range {b} failed")

with ThreadPoolExecutor(parts) as pool:
    blob = b"".join(pool.map(get, bounds))
assert hashlib.sha256(blob).hexdigest() == expected, "checksum mismatch"
(root / rel).write_bytes(blob)
print(f"OK {rel}: {size/1e6:.1f} MB, sha256 verified")
