"""Плакат A0 из шаблона: шапка (команда, авторы, ментор, статья) + две колонки карточек с текстом, формулами и
рисунками. Высоты блоков считаются заранее, чтобы ничего не вылезало за карточки.

    python poster/prepare_assets.py <клон репозитория>   # картинки и формулы
    python poster/build_poster.py --team 07                # -> poster/Команда_07_Постер.pptx
"""
import copy
import re
import sys
from pathlib import Path

from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TEMPLATE = HERE / "template.pptx"

CM = 360000
FONT = "Inter"
ACCENT = RGBColor(0x43, 0x4B, 0x9A)
TEXT = RGBColor(0x1F, 0x22, 0x33)
MUTED = RGBColor(0x5B, 0x60, 0x75)
CARD_LINE = RGBColor(0xC9, 0xCC, 0xE6)
LIGHT = RGBColor(0xF1, 0xF2, 0xFA)

BODY, SMALL, HEAD = 16, 14, 28  # кегли, pt
PAD = 0.7                       # поля карточки, см
GAP = 0.3                      # между блоками внутри карточки, см
CARD_GAP = 0.6                  # между карточками, см
X0, COLW, COLGAP = 6.6, 34.7, 1.5
Y_TOP, Y_BOTTOM = 28.9, 111.6

# --------------------------------------------------------------------------- измерение текста
# Inter в системе нет: меряем Segoe UI. По экспорту плаката Inter шире на 7–8% (жирный — на 4–7%),
# межстрочье PowerPoint для Inter — 1.37 кегля при множителе 1.1; берём с запасом
_FONTS = {False: "C:/Windows/Fonts/segoeui.ttf", True: "C:/Windows/Fonts/segoeuib.ttf"}
WIDTH_SAFETY = 1.11
LINE = 1.1  # множитель межстрочного интервала в pptx


def font(bold):
    try:
        return ImageFont.truetype(_FONTS[bold], 100)
    except OSError:  # не Windows: шрифт из matplotlib (DejaVu шире Segoe — расчёт выйдет с запасом)
        import matplotlib
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), 100)


def runs_of(text):
    """'обычный **жирный** текст' -> [(кусок, жирный?)]"""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    return [(p[2:-2], True) if p.startswith("**") else (p, False) for p in parts if p]


def text_width_cm(text, size_pt, bold):
    return font(bold).getlength(text) / 100 * size_pt * 0.03528 * WIDTH_SAFETY


def count_lines(text, size_pt, width_cm, bold_all=False):
    words = []
    for chunk, bold in runs_of(text):
        for w in re.split(r"(\s+)", chunk):
            if w:
                words.append((w, bold or bold_all))
    lines, current = 1, 0.0
    for w, bold in words:
        wcm = text_width_cm(w, size_pt, bold)
        if w.isspace():
            current += wcm
            continue
        if current + wcm > width_cm and current > 0:
            lines += 1
            current = wcm
        else:
            current += wcm
    return lines


def line_cm(size_pt):
    return size_pt * 0.03528 * 1.29 * LINE  # 1.29 × 1.1 ≈ 1.42 кегля: запас к реальным 1.37


# --------------------------------------------------------------------------- текстовые блоки в pptx
def add_textbox(slide, x, y, w, h):
    box = slide.shapes.add_textbox(Emu(int(x * CM)), Emu(int(y * CM)), Emu(int(w * CM)), Emu(int(h * CM)))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    return tf


def fill_paragraph(p, text, size, color=TEXT, bold_all=False, align=PP_ALIGN.LEFT, space_after=0.0):
    p.alignment = align
    p.line_spacing = LINE
    p.space_after = Pt(space_after / 0.03528) if space_after else Pt(0)
    for chunk, bold in runs_of(text):
        r = p.add_run()
        r.text = chunk
        r.font.size = Pt(size)
        r.font.name = FONT
        r.font.bold = bold or bold_all
        r.font.color.rgb = ACCENT if (bold and not bold_all and color == TEXT) else color


def set_bullet(p, indent_cm=0.75):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(indent_cm * CM)))
    pPr.set("indent", str(int(-indent_cm * CM)))
    for tag in ("a:buNone", "a:buChar", "a:buFont", "a:buClr"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", "434B9A")
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "•")


# --------------------------------------------------------------------------- блоки карточек
class Para:
    def __init__(self, text, size=BODY, color=TEXT, align=PP_ALIGN.LEFT, bold=False):
        self.text, self.size, self.color, self.align, self.bold = text, size, color, align, bold

    def height(self, w):
        return count_lines(self.text, self.size, w, self.bold) * line_cm(self.size)

    def draw(self, slide, x, y, w):
        h = self.height(w)
        tf = add_textbox(slide, x, y, w, h)
        fill_paragraph(tf.paragraphs[0], self.text, self.size, self.color, self.bold, self.align)
        return h


class Bullets:
    INDENT = 0.75
    SPACE = 0.22

    def __init__(self, items, size=BODY, numbered=False):
        self.items, self.size, self.numbered = items, size, numbered

    def height(self, w):
        lines = sum(count_lines(t, self.size, w - self.INDENT) for t in self.items)
        return lines * line_cm(self.size) + self.SPACE * (len(self.items) - 1)

    def draw(self, slide, x, y, w):
        h = self.height(w)
        tf = add_textbox(slide, x, y, w, h + 0.3)
        for i, item in enumerate(self.items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            text = f"**{i + 1}.** {item}" if self.numbered else item
            fill_paragraph(p, text, self.size, space_after=self.SPACE)
            if not self.numbered:
                set_bullet(p, self.INDENT)
        return h


class Picture:
    def __init__(self, path, width=1.0, caption=None, crop=None):
        self.path, self.width, self.caption, self.crop = Path(path), width, caption, crop
        if crop:  # (верх, низ) в долях высоты: берём полосу картинки
            im = Image.open(self.path)
            top, bottom = crop
            self.path = ASSETS / f"_crop_{self.path.stem}.png"
            im.crop((0, int(top * im.height), im.width, int(bottom * im.height))).save(self.path)
        self.w_px, self.h_px = Image.open(self.path).size

    def size(self, w):
        pw = w * self.width
        return pw, pw * self.h_px / self.w_px

    def height(self, w):
        h = self.size(w)[1]
        if self.caption:
            h += 0.25 + count_lines(self.caption, SMALL, w) * line_cm(SMALL)
        return h

    def draw(self, slide, x, y, w):
        pw, ph = self.size(w)
        slide.shapes.add_picture(str(self.path), Emu(int((x + (w - pw) / 2) * CM)), Emu(int(y * CM)),
                                 Emu(int(pw * CM)), Emu(int(ph * CM)))
        if self.caption:
            Para(self.caption, SMALL, MUTED, PP_ALIGN.CENTER).draw(slide, x, y + ph + 0.25, w)
        return self.height(w)


class Formulas:
    """Одна или несколько формул в строку, общая высота строки (см)."""

    def __init__(self, names, height=1.9, gap=2.5):
        self.names, self.h, self.gap = names, height, gap
        self.sizes = [Image.open(ASSETS / f"formula_{n}.png").size for n in names]

    def widths(self, w):
        ws = [self.h * a / b for a, b in self.sizes]
        total = sum(ws) + self.gap * (len(ws) - 1)
        scale = min(1.0, w / total)
        return [x * scale for x in ws], scale

    def height(self, w):
        return self.h * self.widths(w)[1]

    def draw(self, slide, x, y, w):
        ws, scale = self.widths(w)
        h = self.h * scale
        cx = x + (w - sum(ws) - self.gap * scale * (len(ws) - 1)) / 2
        for name, fw in zip(self.names, ws):
            slide.shapes.add_picture(str(ASSETS / f"formula_{name}.png"), Emu(int(cx * CM)), Emu(int(y * CM)),
                                     Emu(int(fw * CM)), Emu(int(h * CM)))
            cx += fw + self.gap * scale
        return h


class Table:
    """Таблица с автоподбором ширины столбцов: каждый столбец не уже своего заголовка и значений (по метрикам
    Inter), чтобы слова не переносились посреди. Не помещается — уменьшаем кегль. Заголовки переносятся только
    там, где стоит «\\n». groups — строка групп над заголовками: [(подпись, сколько столбцов), ...]."""
    MARGIN = 0.18
    SIZES = [(12, 14), (11, 13), (10, 12)]  # (заголовок, ячейки), pt

    def __init__(self, header, rows, col_weights=None, size=17, head_size=15, highlight_last=True, groups=None):
        self.header, self.rows, self.groups, self.hl = header, rows, groups, highlight_last
        self.head_lines = max(h.count("\n") + 1 for h in header)
        self._cache = {}

    def need(self, j, head_size, size):
        texts = [(line, head_size, True) for line in self.header[j].split("\n")]
        texts += [(row[j], size, self.hl and k == len(self.rows) - 1) for k, row in enumerate(self.rows)]
        return max(text_width_cm(t, sz, b) for t, sz, b in texts) + 2 * self.MARGIN + 0.15

    def layout(self, w):
        if w not in self._cache:
            for head_size, size in self.SIZES:
                need = [self.need(j, head_size, size) for j in range(len(self.header))]
                if self.groups:  # подпись группы должна поместиться над своими столбцами
                    col = 0
                    for label, span in self.groups:
                        width = text_width_cm(label, head_size, True) + 2 * self.MARGIN + 0.2
                        have = sum(need[col:col + span])
                        if label and width > have:
                            for k in range(col, col + span):
                                need[k] += (width - have) / span
                        col += span
                if sum(need) <= w:
                    break
            extra = w - sum(need)
            widths = [need[0] + max(extra, 0) * 0.4] + [n + max(extra, 0) * 0.6 / (len(need) - 1) for n in need[1:]]
            scale = w / sum(widths)
            self._cache[w] = ([x * scale for x in widths], head_size, size)
        return self._cache[w]

    def heights(self, w):
        _, head_size, size = self.layout(w)
        row_h = line_cm(size) + 0.22
        head_h = self.head_lines * line_cm(head_size) + 0.4
        group_h = line_cm(head_size) + 0.25 if self.groups else 0.0
        return group_h, head_h, row_h

    def height(self, w):
        group_h, head_h, row_h = self.heights(w)
        return group_h + head_h + row_h * len(self.rows)

    def draw(self, slide, x, y, w):
        widths, head_size, size = self.layout(w)
        group_h, head_h, row_h = self.heights(w)
        n_cols = len(self.header)
        top = 1 if self.groups else 0
        n_rows = len(self.rows) + 1 + top
        gf = slide.shapes.add_table(n_rows, n_cols, Emu(int(x * CM)), Emu(int(y * CM)), Emu(int(w * CM)),
                                    Emu(int(self.height(w) * CM)))
        table = gf.table
        tblPr = table._tbl.tblPr
        tblPr.set("firstRow", "0")
        tblPr.set("bandRow", "0")
        for j, cw in enumerate(widths):
            table.columns[j].width = Emu(int(cw * CM))

        def style(cell):
            cell.margin_left = cell.margin_right = Emu(int(self.MARGIN * CM))
            cell.margin_top = cell.margin_bottom = Emu(int(0.06 * CM))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.word_wrap = True
            cell.text_frame.paragraphs[0].text = ""

        if self.groups:  # строка групп: «Наш результат» / «Статья» над своими столбцами
            table.rows[0].height = Emu(int(group_h * CM))
            col = 0
            for label, span in self.groups:
                cell = table.cell(0, col)
                if span > 1:
                    cell.merge(table.cell(0, col + span - 1))
                style(cell)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0x2E, 0x34, 0x73) if label else RGBColor(255, 255, 255)
                if label:
                    fill_paragraph(cell.text_frame.paragraphs[0], label, head_size, RGBColor(255, 255, 255),
                                   bold_all=True, align=PP_ALIGN.CENTER)
                col += span
        for i in range(top, n_rows):
            r = i - top  # 0 — строка заголовков
            table.rows[i].height = Emu(int((head_h if r == 0 else row_h) * CM))
            for j in range(n_cols):
                cell = table.cell(i, j)
                style(cell)
                text = self.header[j] if r == 0 else self.rows[r - 1][j]
                last = self.hl and i == n_rows - 1
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT if r == 0 else (RGBColor(0xE3, 0xE5, 0xF4) if last else
                                                                  (LIGHT if r % 2 == 0 else RGBColor(255, 255, 255)))
                lines = text.split("\n")
                if r == 0:  # заголовки одной высоты: однострочные опускаем к низу, как двухстрочные
                    lines = [""] * (self.head_lines - len(lines)) + lines
                for k, line in enumerate(lines):
                    para = cell.text_frame.paragraphs[0] if k == 0 else cell.text_frame.add_paragraph()
                    fill_paragraph(para, line or " ", head_size if r == 0 else size,
                                   RGBColor(255, 255, 255) if r == 0 else TEXT, bold_all=(r == 0 or last),
                                   align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
        return self.height(w)


class Row:
    """Картинка (или любой блок) слева и колонка блоков справа."""

    def __init__(self, left, right, split=0.45, gap=0.6):
        self.left, self.right, self.split, self.gap = left, right, split, gap

    def widths(self, w):
        wl = (w - self.gap) * self.split
        return wl, w - self.gap - wl

    def right_h(self, wr):
        return sum(b.height(wr) for b in self.right) + GAP * (len(self.right) - 1)

    def height(self, w):
        wl, wr = self.widths(w)
        return max(self.left.height(wl), self.right_h(wr))

    def draw(self, slide, x, y, w):
        wl, wr = self.widths(w)
        h = self.height(w)
        self.left.draw(slide, x, y, wl)
        cy = y + (h - self.right_h(wr)) / 2  # текст по центру высоты картинки
        for b in self.right:
            cy += b.draw(slide, x + wl + self.gap, cy, wr) + GAP
        return h


class Spacer:
    def __init__(self, h):
        self.h = h

    def height(self, w):
        return self.h

    def draw(self, slide, x, y, w):
        return self.h


# --------------------------------------------------------------------------- карточки и колонки
class Card:
    def __init__(self, title, blocks):
        self.title, self.blocks = title, blocks

    def head_h(self, w):
        return count_lines(self.title, HEAD, w, True) * line_cm(HEAD) + 0.45

    def height(self, w):
        inner = w - 2 * PAD
        return 2 * PAD + self.head_h(inner) + sum(b.height(inner) for b in self.blocks) + GAP * (len(self.blocks) - 1)

    def draw(self, slide, x, y, w, h=None):
        h = h or self.height(w)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(int(x * CM)), Emu(int(y * CM)),
                                      Emu(int(w * CM)), Emu(int(h * CM)))
        card.adjustments[0] = 0.025
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(255, 255, 255)
        card.line.color.rgb = CARD_LINE
        card.line.width = Pt(2)
        card.shadow.inherit = False
        inner = w - 2 * PAD
        cy = y + PAD
        Para(self.title, HEAD, ACCENT, bold=True).draw(slide, x + PAD, cy, inner)
        hh = self.head_h(inner)
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int((x + PAD) * CM)), Emu(int((cy + hh - 0.35) * CM)),
                                      Emu(int(6 * CM)), Emu(int(0.12 * CM)))
        line.fill.solid()
        line.fill.fore_color.rgb = ACCENT
        line.line.fill.background()
        cy += hh
        for b in self.blocks:
            cy += b.draw(slide, x + PAD, cy, inner) + GAP


def layout_column(slide, cards, x, w, y0, y1):
    heights = [c.height(w) for c in cards]
    used = sum(heights) + CARD_GAP * (len(cards) - 1)
    free = (y1 - y0) - used
    print(f"колонка x={x:.1f}: занято {used:.1f} из {y1 - y0:.1f} см, свободно {free:.1f} см")
    if free < -0.01:
        raise SystemExit(f"Колонка не помещается: не хватает {-free:.1f} см")
    extra = free / len(cards)  # свободное место делим между карточками: колонки заканчиваются ровно
    y = y0
    for c, h in zip(cards, heights):
        c.draw(slide, x, y, w, h + extra)
        y += h + extra + CARD_GAP


# --------------------------------------------------------------------------- шапка шаблона
AUTHORS = ["Артём Илларионов, ИТМО", "Алиса Попова, МФТИ", "Андрей Москович, СПбГУ", "Ян Мусатов, ВШЭ",
           "Владимир Турундаев, ВШЭ", "Артём Тучков, Иннополис"]
MENTOR, COMPANY = "Виктор Киселёв", "Сбер"
TEAM = "КОМАНДА № {}"
PAPER = ("Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a "
         "consumer wearable device")
PAPER_REF = "O. Walch, Y. Huang, D. Forger, C. Goldstein. SLEEP, 2019"


def shape_by_name(slide, name):
    for s in slide.shapes:
        if s.name == name:
            return s
    raise KeyError(name)


def set_text_keep_format(shape, text):
    p = shape.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in shape.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def fill_header(slide):
    set_text_keep_format(shape_by_name(slide, "Номер команды"), TEAM.format(TEAM_NUMBER))
    names = [shape_by_name(slide, f"Автор {i} — имя, фамилия, вуз") for i in range(1, 5)]
    marks = [shape_by_name(slide, f"Маркер автора {i}") for i in range(1, 5)]
    for i in range(4, len(AUTHORS)):  # в шаблоне 4 места: копируем последнее
        for src, lst in ((names[3], names), (marks[3], marks)):
            el = copy.deepcopy(src._element)
            src._element.addnext(el)
            lst.append(slide.shapes[list(slide.shapes._spTree.iterchildren()).index(el) - 2])
    top, pitch = 12.75, 1.72
    for i, (n, m, text) in enumerate(zip(names, marks, AUTHORS)):
        n.top = Emu(int((top + i * pitch) * CM))
        m.top = Emu(int((top + i * pitch + 0.72) * CM))
        n.name, m.name = f"Автор {i + 1} — имя, фамилия, вуз", f"Маркер автора {i + 1}"
        set_text_keep_format(n, text)
        for r in n.text_frame.paragraphs[0].runs:
            r.font.size = Pt(36)
    set_text_keep_format(shape_by_name(slide, "Имя и фамилия ментора"), MENTOR)
    set_text_keep_format(shape_by_name(slide, "Компания ментора"), COMPANY)
    title = shape_by_name(slide, "Название статьи")
    set_text_keep_format(title, PAPER)
    title.height = Emu(int(7.5 * CM))
    p0 = title.text_frame.paragraphs[0]
    for r in p0.runs:
        r.font.size = Pt(30)
    p = title.text_frame.add_paragraph()
    r = p.add_run()
    r.text = PAPER_REF
    r.font.size, r.font.name, r.font.italic, r.font.color.rgb = Pt(26), FONT, True, MUTED


# --------------------------------------------------------------------------- содержание
FOOTER_REFS = ("[1] Walch O. et al. Sleep stage prediction with raw acceleration and photoplethysmography heart "
               "rate data derived from a consumer wearable device. SLEEP, 2019   [2] Forger D. et al. A simpler model "
               "of the human circadian pacemaker. J. Biol. Rhythms, 1999   [3] Lundberg S., Lee S. A unified approach "
               "to interpreting model predictions. NeurIPS, 2017")
POSTER_TITLE = "Сон или бодрствование по Apple Watch: воспроизведение статьи, важность признаков и улучшение модели"
REPO_URL = "github.com/AlisaPopova25/ai_school_october"

def shap_fast_picture():
    """SHAP расширенного MLP (6 признаков) от команды: assets/shap_fast_right.png (одна панель) или правая
    половина assets/shap_fast.png (две панели рядом). Нет картинки — None."""
    if (ASSETS / "shap_fast_right.png").exists():
        path = ASSETS / "shap_fast_right.png"
    elif (ASSETS / "shap_fast.png").exists():
        im = Image.open(ASSETS / "shap_fast.png").convert("RGB")
        path = ASSETS / "_shap_fast_right.png"
        im.crop((int(im.width * 0.5), int(im.height * 0.12), im.width, im.height)).save(path)
    else:
        print("нет assets/shap_fast_right.png — карточка 6 без SHAP расширенной модели")
        return None
    return Picture(path, caption="SHAP расширенного MLP (график команды)")


def team_shap_row():
    """SHAP расширенного MLP слева; справа — описание трёх показателей словами и их формулы столбиком."""
    text = [Bullets([
        "Лучше всего было когда к исходным признакам добавлены три показателя:",
        "Пик движения: максимальные колебания ускорения на 5-секундном отрезке внутри текущей эпохи.",
        "Доля движения: доля активных 5-секундных отрезков в минутном окне — текущая эпоха и по 15 секунд до и "
        "после неё.",
        "Относительный пульс: отклонение среднего пульса за минуту от медианы записи, делённое на медиану.",
    ]), Formulas(["fpeak"], height=1.5), Formulas(["factive"], height=1.5), Formulas(["fhr"], height=1.5)]
    picture = shap_fast_picture()
    if picture is None:
        return text
    return [Row(picture, text, split=0.38)]


def experiment_conclusions():
    bullets = Bullets([
        "**Лучше всего — исходные признаки + 3 быстрых:** κ 0.496 → 0.534, AUC 0.908 → 0.917. Быстрые признаки "
        "ловят короткие движения и учащение пульса, которые сглаженный motion «размазывает» по 10 минутам",
        "**Помогают** соседние эпохи и статистики короткого окна — контекст соседних минут. **Не помогают** "
        "повтор примеров бодрствования (переобучение, AUC падает) и разности эпох без сглаживания (шум)",
        "**SHAP расширенной модели:** доля движения за минуту (|SHAP| 0.024) работает почти как motion (0.034); пик "
        "движения и относительный пульс слабее (0.009 и 0.005); время ночи остаётся самым сильным признаком (0.049)",
        "**Ограничение:** окно заглядывает на 15 с вперёд, а медиана пульса берётся по всей записи — это анализ "
        "готовой записи, а не предсказание в реальном времени"])
    picture = shap_fast_picture()
    if picture is None:
        return [Bullets(bullets.items[:2] + bullets.items[3:])]
    return [Row(picture, [bullets], split=0.42)]


LEFT = [
    Card("1. Статья и задача", [
        Para("Полисомнография (PSG) — эталон измерения сна, но нужны лаборатория, датчики на голове и специалист, "
             "поэтому сон измеряют одну-две ночи. **Можно ли размечать сон по смарт-часам — каждую ночь и дома?**"),
        Bullets([
            "**Что сделали авторы:** своё приложение для Apple Watch впервые дало сырые данные ускорения (50 раз в "
            "секунду) и пульса (≈ раз в 5 с) одновременно с PSG; данные и код открыты",
            "**Данные:** 31 человек (21 женщина, средний возраст 29 лет): ночь в лаборатории с PSG и часами + "
            "неделя с часами дома до неё (по шагам оценивается режим дня)",
            "**Задача:** ночь режется на эпохи по 30 с; для каждой по данным часов нужно угадать стадию, которую "
            "по PSG поставил специалист. Две постановки: **сон / бодрствование** и **бодрствование / NREM / REM** "
            "(NREM — неглубокий и глубокий сон N1–N3, REM — сон с быстрыми движениями глаз). Из предсказаний "
            "собираются гипнограмма и показатели ночи: время сна, засыпания, пробуждений",
            "**Результат статьи** — лучшая модель — нейросеть, обученная на признаках motion + hr + clock. Сон / бодрствование: порог "
            "подобран, чтобы найти 93% эпох сна; тогда найдено 59.6% эпох бодрствования и всего верно 90% эпох "
            "(κ = 0.45). Три класса: в лучшей точке верно ≈ 72% эпох. На 188 людях из исследования MESA "
            "(другой прибор и люди) — сопоставимо",
        ]),
        Row(Picture(ASSETS / "night_poster.png",
                    caption="Испытуемый 46343; движение — activity counts (усл. ед.)"),
            [Para("**Как читать:**", SMALL),
             Bullets(["**Движение** во сне почти всегда 0; пики — пробуждения или ворочания",
                      "**Пульс** ровный в глубоком сне N3 и скачет при пробуждениях",
                      "**Стадии** идут циклами ≈ 90 мин: N3 — в начале ночи, REM к утру длиннее",
                      "**Трудное место:** в начале ночи человек не спит, но почти не двигается — по часам это "
                      "бодрствование не видно"], size=SMALL)],
            split=0.55),
    ]),
    Card("2. Признаки: одно значение на каждую эпоху", [
        Bullets([
            "**motion** — сколько человек двигался: из ускорения убираются гравитация и шум, движения суммируются "
            "за 15 с («activity counts», как у медицинских актиграфов) и усредняются за ≈ 10 мин",
            "**hr** — насколько сильно меняется пульс вокруг эпохи: разброс пульса за ≈ 10 мин, без медленного "
            "изменения за ночь; мал в глубоком сне, велик при пробуждениях",
            "**clock** — фаза суточного ритма: шаги за прошлые дни → оценка освещённости → модель внутренних часов "
            "Forger (1999) → волна x(t); ночные данные не нужны",
            "В статье сравнивали также **cosine** (одинаковая для всех волна 24 ч) и **time** (часы от отбоя); "
            "лучший результат — с clock",
        ]),
        Formulas(["clock"], height=1.9),
        Para("x — состояние модели внутренних часов (без единиц). Шкала перевёрнута: clock = 0 — человек лёг, "
             "1 — дно суточного ритма (минимум x, у каждого в своё время); под утро clock снова уменьшается", SMALL, MUTED,
             PP_ALIGN.CENTER),
    ]),
    Card("3. Модели, проверка и метрики", [
        Bullets([
            "**Модели статьи:** логистическая регрессия, kNN (k = 500), случайный лес, нейросеть MLP (3 слоя по 15); "
            "**наша добавка — CatBoost**",
            "**Проверка:** в статье — 50 случайных разбиений людей 70/30; у нас — leave-one-subject-out: каждого из "
            "31 человека предсказывает модель, обученная на остальных 30",
            "**Порог** на P(сон) подбирается так, чтобы найти заданную долю сна (в таблицах — 90%), и смотрим, "
            "сколько найдено бодрствования",
        ]),
        Formulas(["logreg"], height=1.6),
        Bullets([
            "**Accuracy** — доля верных эпох; **Найдено wake** — доля эпох бодрствования, названных бодрствованием "
            "(специфичность); **κ** — согласие с PSG сверх случайного (0 — как монетка, 1 — полное)",
            "**AUC** — качество упорядочивания эпох по всем порогам (0.5 — случайно); **AP wake** — площадь под "
            "кривой precision–recall для бодрствования (случайно — 0.084); **F1** — среднее гармоническое точности "
            "и полноты класса, **Macro F1** — среднее по классам",
        ], size=SMALL),
        Formulas(["kappa", "f1"], height=1.7, gap=3.5),
    ]),
    Card("4. Воспроизведение: совпало со статьёй", [
        Para("**Сон / бодрствование** (признаки motion + hr + clock; порог найдёт 90% сна):", SMALL),
        Table(["Модель", "AUC", "Accuracy", "Найдено\nwake", "κ", "AP\nwake", "F1\nwake", "Macro\nF1",
               "AUC", "Accuracy", "Найдено\nwake", "κ"],
              [["Лог. регрессия", "0.860", "0.876", "0.617", "0.392", "0.479", "0.457", "0.693",
                "0.854", "0.871", "0.556", "0.347"],
               ["kNN", "0.867", "0.880", "0.668", "0.424", "0.502", "0.486", "0.709",
                "0.868", "0.877", "0.627", "0.391"],
               ["Случайный лес", "0.869", "0.880", "0.660", "0.418", "0.562", "0.481", "0.707",
                "0.871", "0.879", "0.653", "0.405"],
               ["Нейросеть (MLP)", "0.877", "0.881", "0.677", "0.429", "0.560", "0.491", "0.712",
                "0.878", "0.881", "0.675", "0.424"],
               ["CatBoost (наш)", "0.880", "0.881", "0.680", "0.431", "0.547", "0.492", "0.713",
                "—", "—", "—", "—"]],
              groups=[("", 1), ("Наш результат (leave-one-subject-out)", 7), ("Статья (Monte Carlo)", 4)]),
        Para("**Три класса** (найдено 60% бодрствования; «найдено NREM / REM» — при уравненных долях, «лучшая "
             "accuracy» и κ — в лучшей по accuracy точке, как в статье):", SMALL),
        Table(["Модель", "Найдено\nNREM", "Найдено\nREM", "Лучшая\naccuracy", "κ", "Macro\nF1", "F1\nREM",
               "Найдено\nNREM", "Найдено\nREM", "Лучшая\naccuracy", "κ"],
              [["Лог. регрессия", "0.645", "0.642", "0.688", "0.175", "0.572", "0.498", "0.623", "0.623", "0.699",
                "0.130"],
               ["kNN", "0.634", "0.637", "0.697", "0.184", "0.571", "0.481", "0.648", "0.647", "0.721", "0.243"],
               ["Случайный лес", "0.633", "0.631", "0.695", "0.249", "0.566", "0.482", "0.638", "0.638", "0.686",
                "0.302"],
               ["Нейросеть (MLP)", "0.643", "0.641", "0.707", "0.228", "0.583", "0.482", "0.651", "0.650", "0.723",
                "0.277"],
               ["CatBoost (наш)", "0.649", "0.654", "0.704", "0.244", "0.586", "0.497", "—", "—", "—", "—"]],
              groups=[("", 1), ("Наш результат (leave-one-subject-out)", 6), ("Статья", 4)]),
        Bullets([
            "**Совпало:** AUC, accuracy и доли найденных стадий — в пределах 0.01–0.03 от статьи при другой схеме "
            "проверки. CatBoost лучший по большинству метрик, но отрыв в тысячных: качество упирается в признаки",
            "**Accuracy обманчива:** бодрствование — 8.4% эпох, и «всегда сон» даёт 0.916 > 0.88 у всех моделей",
            "**Слабость:** найдено ≈ 2/3 бодрствования, но верна лишь каждая третья метка «бодрствование» "
            "(PPV 0.36–0.39): тихое бодрствование без движения не видно. REM — самый трудный (F1 ≈ 0.5)",
        ]),
    ]),
]


def fast_results_blocks():
    """Таблица «без быстрых / с быстрыми признаками» из outputs/repro/fast_features/results.csv
    (python repro/10_fast_features.py), если её уже посчитали."""
    import pandas as pd
    path = HERE.parent / "outputs" / "repro" / "fast_features" / "results.csv"
    if not path.exists():
        return [Para("Те же признаки добавлены к **CatBoost** и нейросети в общей постановке проекта "
                     "(motion, hr, clock; leave-one-subject-out): python repro/10_fast_features.py — после "
                     "запуска таблица появится здесь", SMALL)]
    df = pd.read_csv(path).set_index(["model", "feature_set"])
    cols = ["auc", "ap_wake", "wake_correct@90", "kappa@90", "f1_wake@90"]
    rows = []
    for model, label in [("CatBoost", "CatBoost")]:  # на плакате — только CatBoost с clock
        for base in ["Clock"]:
            key = (model, f"Motion, HR, {base}")
            if key not in df.index:
                continue
            plain, fast = df.loc[key], df.loc[(model, f"Motion, HR, {base} + быстрые")]
            rows.append([f"{label} · {base.lower()}"] + [f"{plain[c]:.3f}" for c in cols]
                        + [f"{fast[c]:.3f}" for c in cols])
    cb = df.loc[("CatBoost", "Motion, HR, Clock")], df.loc[("CatBoost", "Motion, HR, Clock + быстрые")]
    head = ["Модель", "AUC", "AP\nwake", "Найдено\nwake", "κ", "F1\nwake",
            "AUC", "AP\nwake", "Найдено\nwake", "κ", "F1\nwake"]
    return [
        Para("**Те же признаки на CatBoost** — общая постановка проекта: leave-one-subject-out, признаки motion + hr + "
             "clock, порог найдёт 90% сна:", SMALL),
        Table(head, rows, highlight_last=False,
              groups=[("", 1), ("Без быстрых признаков", 5), ("С быстрыми признаками", 5)]),
        Bullets([
            f"**Быстрые признаки улучшают CatBoost по всем метрикам:** AUC "
            f"{cb[0]['auc']:.3f} → **{cb[1]['auc']:.3f}** — лучший результат проекта; найдено бодрствования "
            f"{100 * cb[0]['wake_correct@90']:.0f}% → {100 * cb[1]['wake_correct@90']:.0f}%, AP wake "
            f"{cb[0]['ap_wake']:.3f} → {cb[1]['ap_wake']:.3f}, κ {cb[0]['kappa@90']:.3f} → {cb[1]['kappa@90']:.3f}",
        ], size=SMALL),
    ]


RIGHT = [
    Card("5. Какие признаки важны: значения Шепли", [
        Formulas(["shapley"], height=2.0),
        Bullets([
            "**Идея:** признаки — «игроки», качество модели — «выигрыш». Модель переобучается на всех 7 непустых "
            "подмножествах {motion, hr, clock}; вклад признака φᵢ — его средний прирост метрики по всем порядкам "
            "добавления. Отсчёт — от случайной модели (AUC 0.5), сумма вкладов = качество полной модели − 0.5",
            "**Почему не просто «качество одного признака»:** признаки частично дублируют друг друга (при движении "
            "растёт и пульс). Motion, добавленный первым, даёт +0.29 AUC, а последним — лишь +0.05. Шепли делит "
            "общую информацию поровну",
            "**Результат:** motion — около половины прироста, clock — четверть–треть, hr — меньше всех; картина "
            "одинакова у всех пяти моделей. Для поиска бодрствования (AP wake) доля пульса падает до 12%: он "
            "помогает упорядочить эпохи в целом, но почти не помогает точно находить бодрствование",
            "**SHAP по эпохам** (вклад признака в P(сон) конкретной эпохи): покой → сон, движение → бодрствование, "
            "начало ночи → бодрствование, ровный пульс → немного к сну",
        ]),
        Picture(ASSETS / "shapley_poster.png", width=0.82),
    ]),
    Card("6. Улучшение модели: быстрые признаки", [  # текст и раскладка — из правок команды в PowerPoint
        Para("Эксперимент команды MLP: исследованы соседние эпохи, статистики и изменения сигналов, балансировка "
             "классов и подбор порога."),
        Para("Лучший результат дало добавление трёх локальных признаков движения и пульса:"),
        Para("Cohen’s κ выросла с 0,496 до 0,534, ROC AUC — с 0,908 до 0,917 при проверке с разделением по "
             "участникам."),
        Picture(ASSETS / "experiments.png", width=0.78),
    ] + team_shap_row() + fast_results_blocks()),
]

CONCLUSIONS = Card("Выводы", [
    Bullets([
        "**Статья воспроизводится.** На открытых данных и современном Python AUC моделей (0.86–0.88), accuracy и доли "
        "найденных стадий совпали с авторскими в пределах 0.01–0.03, хотя мы проверяли строже: каждого человека "
        "предсказывала модель, которая его не видела. Значит, результат статьи переносится на новых людей",
        "**Модель почти не важна — важны признаки.** Пять моделей, от логистической регрессии до CatBoost, "
        "различаются по AUC лишь на 0.02. CatBoost лучший (AUC 0.880), но отрыв от нейросети статьи — в тысячных",
        "**Главное — движение и время ночи.** По Шепли около половины прироста качества даёт движение, "
        "четверть–треть — clock, пульс — меньше всего, а для поиска бодрствования лишь 12–18%. Картина одинакова "
        "у всех моделей, то есть это свойство данных, а не модели",
        "**Accuracy скрывает главную слабость.** «Всегда сон» даёт 0.916 — больше, чем 0.88 у моделей. Модели "
        "находят около 2/3 бодрствования, но верна лишь каждая третья метка «бодрствование»: тихое бодрствование "
        "без движения по часам не видно. REM — самый трудный класс (F1 ≈ 0.5). Поэтому оцениваем по κ, AP и F1",
        "**Быстрые признаки помогают всем моделям.** Пик движения, доля движения за минуту и относительный пульс "
        "по 5-секундным отрезкам подняли κ нейросети команды с 0.496 до 0.534, а у CatBoost в общей постановке — "
        "AUC с 0.880 до 0.890 (лучший результат проекта) и долю найденного бодрствования с 68% до 71%",
        "**Что дальше:** признаки, которые видят тихое бодрствование и REM (дыхание и вариабельность пульса по "
        "сырому сигналу PPG), учёт последовательности эпох во времени, проверка на других людях и устройствах "
        "(MESA) и вариант быстрых признаков без «заглядывания» на 15 с вперёд — для работы в реальном времени",
    ], numbered=True),
])


def columns():
    return LEFT, RIGHT + [CONCLUSIONS]


TEAM_NUMBER = "[00]"


def main():
    global TEAM_NUMBER
    import argparse
    parser = argparse.ArgumentParser(description="Собрать плакат A0 из шаблона")
    parser.add_argument("--team", help="номер команды; файл будет называться Команда_XX_Постер.pptx")
    args = parser.parse_args()
    if args.team:
        TEAM_NUMBER = args.team
    out = HERE / (f"Команда_{args.team}_Постер.pptx" if args.team else "poster.pptx")
    prs = Presentation(TEMPLATE)
    slide = prs.slides[0]
    fill_header(slide)
    tf = add_textbox(slide, 18.6, 24.0, 59.0, 4.2)
    fill_paragraph(tf.paragraphs[0], POSTER_TITLE, 42, ACCENT, bold_all=True, align=PP_ALIGN.CENTER)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    left, right = columns()
    layout_column(slide, left, X0, COLW, Y_TOP, Y_BOTTOM)
    layout_column(slide, right, X0 + COLW + COLGAP, COLW, Y_TOP, Y_BOTTOM)
    foot = add_textbox(slide, X0, Y_BOTTOM + 0.6, 2 * COLW + COLGAP, 2.4)
    fill_paragraph(foot.paragraphs[0], FOOTER_REFS, 17, MUTED, align=PP_ALIGN.CENTER)
    fill_paragraph(foot.add_paragraph(), "AI360 · Проектная школа · 2026", 19, MUTED, align=PP_ALIGN.CENTER)
    try:
        prs.save(out)
    except PermissionError:  # файл открыт в PowerPoint: пишем рядом следующую свободную версию
        base = out
        for k in range(2, 100):
            out = base.with_name(f"{base.stem}_v{k}{base.suffix}")
            try:
                prs.save(out)
                break
            except PermissionError:
                continue
    print("записано:", out)


if __name__ == "__main__":
    main()
