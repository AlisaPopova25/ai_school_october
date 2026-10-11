"""Прогресс-бар без зависимостей: этап, процент, прошло и сколько осталось (оценка по средней скорости).

    bar = Progress(total=248, title="обучение")
    bar.message("== модель 2/5: Random Forest")      # строка-этап, остаётся в выводе
    bar.update(status="Motion, HR · 3 класса")      # +1 шаг, полоса перерисовывается на месте
    bar.close()

Где показывается:
  * в терминале — одна строка, обновляется на месте («\\r»), обрезается по ширине окна;
  * под run_all.py / train_catboost.py (SLEEP_PROGRESS=1) — полоса уходит родителю строками с маркером MARK
    (не через «\\r»: в текстовом режиме канала Python превращает «\\r» в «\\n»), родитель рисует её сам;
  * в остальных случаях (вывод в файл) — отдельной строкой не чаще раза в 30 с.
"""
import os
import shutil
import sys
import time

MARK = "\x01progress\x01"            # промежуточное состояние полосы: родитель рисует его на месте
MARK_DONE = "\x01progress-done\x01"  # итог полосы: родитель оставляет его в выводе и в логе


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def safe_output(stream=None):
    """Не падать на символах, которых нет в кодировке вывода (cp1251 при перенаправлении в файл на Windows)."""
    stream = stream or sys.stdout
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")


def bar_chars(stream):
    try:
        "█░".encode(getattr(stream, "encoding", None) or "ascii")
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


def terminal_width():
    return max(40, shutil.get_terminal_size((120, 20)).columns - 1)


class Progress:
    def __init__(self, total, title="", width=24, stream=None):
        self.total = max(1, int(total))
        self.title = title
        self.width = width
        self.done = 0
        self.started = time.time()
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.piped = not self.tty and os.environ.get("SLEEP_PROGRESS") == "1"
        self.full, self.empty = bar_chars(self.stream)
        self._last_print = 0.0
        self.status = ""

    def line(self):
        frac = self.done / self.total
        filled = int(round(frac * self.width))
        elapsed = time.time() - self.started
        eta = fmt_time(elapsed / self.done * (self.total - self.done)) if self.done else "…"
        text = (f"{self.title} [{self.full * filled}{self.empty * (self.width - filled)}] {100 * frac:5.1f}% "
                f"{self.done}/{self.total} | прошло {fmt_time(elapsed)} | осталось ~{eta}")
        return text + (f" | {self.status}" if self.status else "")

    def show(self, force=False):
        if self.tty:
            cols = terminal_width()
            self.stream.write("\r" + self.line()[:cols].ljust(cols))
        elif self.piped:
            self.stream.write(MARK + self.line() + "\n")
        elif force or time.time() - self._last_print >= 30:
            self.stream.write(self.line() + "\n")
            self._last_print = time.time()
        self.stream.flush()

    def update(self, n=1, status=None):
        self.done = min(self.total, self.done + n)
        if status is not None:
            self.status = status
        self.show(force=self.done == self.total)

    def message(self, text):
        """Строка-этап: остаётся в выводе над полосой."""
        if self.tty:
            self.stream.write("\r" + " " * terminal_width() + "\r" + text + "\n")
        elif self.piped:
            self.stream.write(text + "\n")
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
        self.show(force=True)

    def close(self):
        if self.piped:
            self.stream.write(MARK_DONE + self.line() + "\n")
        else:
            self.show(force=True)
            if self.tty:
                self.stream.write("\n")
        self.stream.flush()
