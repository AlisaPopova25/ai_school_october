"""Дамп ВСЕХ результатов проекта в один читаемый Markdown-файл (таблицы вместо LaTeX).

    python repro/08_results_markdown.py                      # из рабочих папок: outputs/logs + outputs/repro
    python repro/08_results_markdown.py --snapshot           # из снимка в репозитории: results/
    python repro/08_results_markdown.py --out MY_RESULTS.md

Что попадает в файл:
  1. таблицы статьи (2–6 и F1), которые оригинальный код печатает в LaTeX, — из логов, переведённые в таблицы;
     одинаковые таблицы из разных логов печатаются один раз;
  2. все CSV / JSON нашего анализа: данные, важность признаков, логистическая регрессия, улучшения,
     качество по времени ночи, F1;
  3. список картинок со ссылками.
Числа округляются до 3 знаков. Ничего не пересчитывается — только собирается то, что уже посчитано.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILE_TITLES = {  # имя лога (без расширения и префикса) -> понятное название
    "mc_sw": "Сон/бодрствование, Monte Carlo (рис. 2–3, табл. 2–5)",
    "mc_3class": "Wake / NREM / REM, Monte Carlo (рис. 4, табл. 6)",
    "time_features": "Clock / cosine / time (рис. S4)",
    "loo_sw": "Leave-one-out, сон/бодрствование",
}
HEADER_NAMES = {
    "Accuracy": "Accuracy", "Wake correct (specificity)": "Wake найден (specificity)",
    "Sleep correct (sensitivity)": "Сон найден (sensitivity)", "$\\kappa$": "κ", "AUC": "AUC",
    "Wake Correct": "Wake найден", "NREM Correct": "NREM найден", "REM Correct": "REM найден",
    "Best accuracy": "Лучшая accuracy", "Wake correct": "Wake найден", "F1 wake": "F1 wake",
    "F1 sleep": "F1 sleep", "Macro F1": "Macro F1", "F1 NREM": "F1 NREM", "F1 REM": "F1 REM",
}


# --------------------------------------------------------------------------- Markdown
def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.3g}"
    return str(value).replace("|", "/").replace("\n", " ")


def md_table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [fmt(c) for c in row] + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(cells[:len(header)]) + " |")
    return "\n".join(lines)


def df_table(df, columns=None, rename=None, max_rows=None):
    if df is None or df.empty:
        return "_нет данных_"
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    if rename:
        df = df.rename(columns=rename)
    if max_rows:
        df = df.head(max_rows)
    return md_table(df.itertuples(index=False), [str(c) for c in df.columns])


# --------------------------------------------------------------------------- LaTeX из логов -> таблицы
def clean(cell):
    cell = cell.replace("\\hline", "").replace("$\\kappa$", "κ").strip()
    return HEADER_NAMES.get(cell, cell)


def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def nice_caption(caption):
    """Английские подписи LaTeX-таблиц статьи -> короткие русские."""
    match = re.search(r"Sleep/wake differentiation performance by (.+?) across", caption)
    if match:
        return f"Сон/бодрствование: accuracy, найденные wake и сон, κ, AUC — {match.group(1)}"
    match = re.search(r"F1 for sleep/wake by (.+?) \(", caption)
    if match:
        return f"F1 для сна/бодрствования (wake — положительный класс) — {match.group(1)}"
    return caption


def parse_latex_tables(text):
    """Таблицы, которые печатает source/analysis/tables/table_builder.py: [(название, заголовок, строки)]."""
    lines = text.splitlines()
    tables, classifier = [], None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Running "):
            classifier = line[len("Running "):].rstrip(". ")
        start = line.find("\\begin{tabular}")
        if start < 0:
            i += 1
            continue
        caption = re.search(r"\\caption\{([^}]*)\}", line)
        caption = caption.group(1) if caption else None
        # спецификация колонок может содержать вложенные {} (например {l*{5}{c}}): заголовок — всё после неё
        spec_match = re.match(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}", line[start:])
        header_text = line[start + spec_match.end():]
        header = [clean(c) for c in header_text.rstrip().rstrip("\\").split("&")]
        body = []
        i += 1
        while i < len(lines) and "\\end{tabular}" not in lines[i]:
            body.append(lines[i])
            i += 1
        rows = []
        for raw in " ".join(body).split("\\\\"):
            raw = raw.replace("\\hline", "").strip()
            if not raw:
                continue
            rows.append([clean(c) for c in raw.split("&")])
        # пустые ведущие ячейки заголовка — подписи строк
        if header and header[0] == "":
            header[0] = "Признаки"
            if len(header) > 1 and header[1] == "":
                header[0], header[1] = "Модель", "Признаки"
        name = nice_caption(caption) if caption else (
            "F1 для трёх классов" if "Macro F1" in header else "Три класса: доли найденных стадий, лучшая accuracy, κ")
        if classifier and classifier not in name:
            name = f"{name} — {classifier}"
        tables.append((name, header, rows))
        i += 1
    return tables


def log_task(path):
    """Имя задачи из имени лога: lr_f1_mc_sw.log -> mc_sw."""
    stem = path.stem
    for prefix in ("catboost_", "random_forest_", "lr_f1_", "logreg_f1_", "lr_", "logreg_", "orig_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def log_title(path):
    return FILE_TITLES.get(log_task(path), path.stem)


# --------------------------------------------------------------------------- секции
def read_csv(path):
    return pd.read_csv(path, dtype={"subject": str}) if path.exists() else None


def section_paper(tables_dir):
    out = ["## 1. Воспроизведение статьи (оригинальный код)",
           "Таблицы, которые код статьи печатает в LaTeX, переведены в Markdown. Monte Carlo по людям 70/30, seed 2019. "
           "«Сон найден» — порог, при котором модель находит заданную долю сна (как в статье). "
           "Если одна и та же таблица есть в нескольких логах, показан самый свежий прогон."]
    seen = set()
    # новые логи первыми: из повторов одной таблицы остаётся самый свежий прогон
    files = sorted((p for p in tables_dir.glob("*") if p.suffix in (".log", ".txt")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    blocks = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\\begin{tabular}" not in text:
            continue
        tables = parse_latex_tables(text)
        new = []
        for name, header, rows in tables:
            # та же таблица = то же название, колонки и подписи строк (числа в прогонах могут чуть отличаться)
            key = (name, tuple(header), tuple(c for r in rows for c in r if c and not is_number(c)))
            if key not in seen:
                seen.add(key)
                new.append((name, header, rows))
        if new:
            blocks.append((path, new))
    order = list(FILE_TITLES)  # в порядке статьи: табл. 2, табл. 6, рис. S4, остальное
    blocks.sort(key=lambda b: (order.index(log_task(b[0])) if log_task(b[0]) in order else len(order), b[0].name))
    found = bool(blocks)
    for path, new in blocks:
        out.append(f"### {log_title(path)}  \n_источник: `{path.name}`_")
        for name, header, rows in new:
            out += [f"**{name}**", md_table(rows, header)]
    if not found:
        out.append("_логов с таблицами не найдено_")
    return out


def section_dataset(adir):
    d = read_csv(adir / "dataset" / "dataset_summary.csv")
    out = ["## 2. Данные (31 испытуемый)"]
    if d is None:
        return out + ["_нет dataset_summary.csv_"]
    agg = d.drop(columns=["subject"]).agg(["mean", "std", "min", "max"]).T.reset_index()
    agg.columns = ["показатель", "среднее", "SD", "мин", "макс"]
    out += ["**Сводка по испытуемым**", df_table(agg),
            "**По каждому испытуемому**",
            df_table(d, ["subject", "psg_hours", "epochs_scored", "motion_hz_night", "hr_median_interval_s_night",
                         "min_Wake", "min_N1", "min_N2", "min_N3", "min_N4", "min_REM"],
                     {"subject": "id", "psg_hours": "ночь, ч", "epochs_scored": "эпох", "motion_hz_night": "ускор., Гц",
                      "hr_median_interval_s_night": "пульс, с", "min_Wake": "Wake, мин", "min_N1": "N1", "min_N2": "N2",
                      "min_N3": "N3", "min_N4": "N4", "min_REM": "REM"})]
    return out


def section_importance(adir):
    a = adir / "importance"
    out = ["## 3. Важность признаков (leave-one-subject-out)"]
    subsets = read_csv(a / "A_subsets.csv")
    if subsets is not None:
        out += ["**Качество на всех подмножествах признаков** (сон/бодрствование)",
                df_table(subsets, ["model", "subset", "auc", "wake_correct@90", "kappa@90", "auc_subject_mean"],
                         {"model": "модель", "subset": "признаки", "wake_correct@90": "wake найден @90% сна",
                          "kappa@90": "κ @90%", "auc_subject_mean": "AUC, среднее по людям"})]
    shap_auc = read_csv(a / "A_shapley_auc.csv")
    if shap_auc is not None:
        out += ["**Аддитивная важность: значение Шепли по AUC**",
                df_table(shap_auc, None, {"model": "модель", "feature": "признак", "shapley_auc": "вклад Шепли в AUC−0.5",
                                          "drop_column_auc": "потеря AUC без признака", "alone_auc": "AUC−0.5 один",
                                          "share": "доля"})]
    weights = a / "B_weights.json"
    if weights.exists():
        w = json.loads(weights.read_text(encoding="utf-8"))
        feats = list(w["lr_sleep_vs_wake"])
        rows = [["LR, коэффициент на 1 SD (sleep vs wake)"] + [w["lr_sleep_vs_wake"][f] for f in feats],
                ["LR, среднее по 31 LOSO-фолду"] + [w["lr_coef_loso_mean"][f] for f in feats],
                ["LR, SD по фолдам"] + [w["lr_coef_loso_std"][f] for f in feats],
                ["Random forest, MDI"] + [w["rf_mdi"][f] for f in feats]]
        rows += [[f"LR 3 класса, класс {c}"] + [w["lr_three_class"][c][f] for f in feats] for c in w["lr_three_class"]]
        out += ["**Веса моделей**", md_table(rows, ["что"] + feats)]
    perm = read_csv(a / "C_permutation.csv")
    if perm is not None:
        out += ["**Permutation importance** (падение AUC при перемешивании признака)",
                df_table(perm, ["model", "feature", "mode", "auc_drop_mean", "auc_drop_std", "base_auc"],
                         {"model": "модель", "feature": "признак", "mode": "как перемешивали",
                          "auc_drop_mean": "падение AUC", "auc_drop_std": "SD", "base_auc": "AUC без перемешивания"})]
    corr = read_csv(a / "features_spearman_corr.csv")
    if corr is not None:
        corr = corr.rename(columns={corr.columns[0]: "признак"})
        out += ["**Корреляции признаков (Spearman)**", df_table(corr)]
    return out


def section_logreg(adir):
    a = adir / "logreg"
    out = ["## 4. Логистическая регрессия: подробный разбор"]
    report = a / "logreg_report.json"
    if report.exists():
        r = json.loads(report.read_text(encoding="utf-8"))
        out.append(f"**Уравнение модели статьи** (подобрано GridSearch: {r['best_params']}):  \n`{r['equation']}`")
        feats = list(r["coef_raw"])
        ci = r["bootstrap_95ci"]
        rows = [[f, r["coef_raw"][f], r["coef_per_SD"][f], r["odds_ratio_per_SD"][f], f"[{ci[f][0]}; {ci[f][1]}]",
                 r["naive_se_iid"][f], r["bootstrap_se"][f]] for f in feats]
        out += [md_table(rows, ["признак", "коэф.", "на 1 SD", "отношение шансов на 1 SD", "95% CI (бутстреп по людям)",
                                "SE «наивная»", "SE бутстреп"])]
        loso = r["loso"]
        out += ["**Качество модели статьи, LOSO**",
                md_table([[k, v] for k, v in loso.items()], ["метрика", "значение"])]
        lin = r.get("linearity_variants", {})
        if lin:
            cols = list(next(iter(lin.values())))
            out += ["**Варианты логистической регрессии (проверка линейности)**",
                    md_table([[name] + [vals[c] for c in cols] for name, vals in lin.items()], ["вариант"] + cols)]
        tc = r.get("three_class", {})
        if tc:
            cols = list(next(iter(tc.values())))
            out += ["**Три класса: one-vs-rest (как в статье) против softmax**",
                    md_table([[name] + [vals[c] for c in cols] for name, vals in tc.items()], ["вариант"] + cols)]
        coefs = r.get("three_class_ovr_coef_raw", {})
        if coefs:
            out += ["**Коэффициенты трёхклассовой модели (one-vs-rest, сырые)**",
                    md_table([[c] + [v[f] for f in feats] for c, v in coefs.items()], ["класс"] + feats)]
    per = read_csv(a / "loso_per_subject.csv")
    if per is not None:
        out += ["**AUC по людям** (модель, обученная без этого человека)",
                df_table(per.sort_values("auc"), None, {"subject": "id", "n": "эпох", "wake_frac": "доля wake"})]
    return out


def section_improve(adir):
    a = adir / "improve"
    out = ["## 5. Улучшение модели (расширенные признаки, LightGBM, HMM)"]
    sw = read_csv(a / "sleep_wake_comparison.csv")
    if sw is not None:
        out += ["**Сон/бодрствование**",
                df_table(sw, ["model", "auc", "auc_subject_median", "wake_correct@90", "accuracy@90", "kappa@90",
                              "wake_correct@93", "kappa@93", "f1_wake@90", "f1_macro@90", "best_f1_wake"],
                         {"model": "модель", "auc_subject_median": "AUC, медиана по людям",
                          "wake_correct@90": "wake найден @90%", "accuracy@90": "accuracy @90%", "kappa@90": "κ @90%",
                          "wake_correct@93": "wake найден @93%", "kappa@93": "κ @93%"})]
    tc = read_csv(a / "three_class_comparison.csv")
    if tc is not None:
        out += ["**Три класса**", df_table(tc, None, {"model": "модель"})]
    abl = read_csv(a / "ablation_groups.csv")
    if abl is not None:
        out += ["**Абляция групп признаков (LightGBM)**", df_table(abl, None, {"groups": "группы", "n_features": "признаков"})]
    shap = a / "shap_lgbm_ext_sw_mean_abs.csv"
    if shap.exists():
        s = pd.read_csv(shap)
        s.columns = ["признак", "mean |SHAP|"]
        out += ["**Топ-15 признаков по SHAP** (LightGBM, расширенные, сон/бодрствование)", df_table(s, max_rows=15)]
    return out


def section_time(adir):
    q = read_csv(adir / "time_of_night" / "quality_by_hour.csv")
    out = ["## 6. Качество по времени ночи"]
    if q is None:
        return out + ["_нет quality_by_hour.csv_"]
    out.append("Один порог на всю ночь (90% сна); recall — доля найденных эпох класса в этом часе.")
    out.append(df_table(q, ["features", "model", "hour", "n", "subjects", "share_wake", "share_rem", "auc_sw",
                            "wake_recall", "sleep_recall", "recall3_wake", "recall3_nrem", "recall3_rem"],
                        {"features": "признаки", "model": "модель", "hour": "час", "n": "эпох", "subjects": "людей",
                         "share_wake": "доля wake", "share_rem": "доля REM", "auc_sw": "AUC", "wake_recall": "wake найден",
                         "sleep_recall": "сон найден", "recall3_wake": "3 кл.: wake", "recall3_nrem": "3 кл.: NREM",
                         "recall3_rem": "3 кл.: REM"}))
    return out


def section_f1(adir):
    a = adir / "f1"
    out = ["## 7. F1-мера"]
    sw = read_csv(a / "f1_sleep_wake.csv")
    if sw is not None:
        out += ["**Сон/бодрствование** (порог: 90% сна найдено; «всегда сон» — модель-пустышка для сравнения)",
                df_table(sw, ["model", "accuracy", "wake_correct", "kappa", "auc", "f1_wake", "f1_sleep", "f1_macro",
                              "best_f1_wake", "best_f1_wake_threshold"],
                         {"model": "модель", "wake_correct": "wake найден", "kappa": "κ",
                          "best_f1_wake": "лучший F1 wake", "best_f1_wake_threshold": "при P(wake) ≥"})]
    tc = read_csv(a / "f1_three_class.csv")
    if tc is not None:
        out += ["**Три класса** (argmax; «всегда NREM» — пустышка)", df_table(tc, None, {"model": "модель", "kappa": "κ"})]
    ps = read_csv(a / "f1_per_subject.csv")
    if ps is not None:
        g = ps.groupby("model", sort=False).f1_wake.describe()[["min", "25%", "50%", "75%", "max"]].reset_index()
        out += ["**F1 бодрствования по людям**", df_table(g, None, {"model": "модель", "50%": "медиана"})]
    return out


# --------------------------------------------------------------------------- все метрики статьи (09_paper_metrics.py)
GLOSSARY = [
    ("Accuracy", "доля всех эпох, распознанных верно; при 92% сна «всегда сон» уже даёт 0.92"),
    ("Сон найден (sensitivity)", "доля эпох сна, названных сном; в статье порог подбирают так, чтобы она была 0.80–0.95"),
    ("Wake найден (specificity)", "доля эпох бодрствования, названных бодрствованием — главное слабое место моделей"),
    ("κ (каппа Коэна)", "согласие с PSG сверх случайного: 0 — как монетка, 1 — полное"),
    ("AUC", "площадь под ROC-кривой: качество разделения сна и бодрствования сразу по всем порогам"),
    ("AP", "площадь под precision–recall кривой для wake (рис. 3): честнее AUC при редком wake"),
    ("PPV wake / PPV сон", "доля верных среди эпох, названных wake / сном (precision); в коде статьи есть, в таблицах нет"),
    ("F1", "гармоническое среднее precision и recall класса; macro F1 — среднее по классам"),
    ("Сдвиг и границы согласия", "Bland–Altman: среднее (PSG − модель) и ±1.96 SD этой разности по людям"),
]
PAPER_SW = {  # Таблицы 2–5 статьи при «сон найден» = 0.9: (AUC, wake найден, κ)
    ("Logistic Regression", "Motion only"): (0.819, 0.549, 0.343), ("Logistic Regression", "HR only"): (0.743, 0.326, 0.187),
    ("Logistic Regression", "Motion, HR"): (0.830, 0.546, 0.341), ("Logistic Regression", "Motion, HR, Clock"): (0.854, 0.556, 0.347),
    ("k-Nearest Neighbors", "Motion only"): (0.803, 0.483, 0.300), ("k-Nearest Neighbors", "HR only"): (0.682, 0.237, 0.117),
    ("k-Nearest Neighbors", "Motion, HR"): (0.810, 0.496, 0.308), ("k-Nearest Neighbors", "Motion, HR, Clock"): (0.868, 0.627, 0.391),
    ("Random Forest", "Motion only"): (0.810, 0.530, 0.329), ("Random Forest", "HR only"): (0.708, 0.282, 0.152),
    ("Random Forest", "Motion, HR"): (0.816, 0.519, 0.322), ("Random Forest", "Motion, HR, Clock"): (0.871, 0.653, 0.405),
    ("Neural Net", "Motion only"): (0.815, 0.542, 0.342), ("Neural Net", "HR only"): (0.737, 0.323, 0.187),
    ("Neural Net", "Motion, HR"): (0.828, 0.528, 0.333), ("Neural Net", "Motion, HR, Clock"): (0.878, 0.675, 0.424),
}
PAPER_T6 = {  # Таблица 6 статьи: (NREM найден, REM найден, лучшая accuracy, κ) при wake найден = 0.6
    ("Logistic Regression", "Motion only"): (0.506, 0.332, 0.710, 0.085), ("Logistic Regression", "HR only"): (0.452, 0.453, 0.698, 0.033),
    ("Logistic Regression", "Motion, HR"): (0.625, 0.625, 0.701, 0.161), ("Logistic Regression", "Motion, HR, Clock"): (0.623, 0.623, 0.699, 0.130),
    ("k-Nearest Neighbors", "Motion only"): (0.294, 0.532, 0.698, 0.072), ("k-Nearest Neighbors", "HR only"): (0.402, 0.402, 0.671, 0.108),
    ("k-Nearest Neighbors", "Motion, HR"): (0.607, 0.605, 0.711, 0.227), ("k-Nearest Neighbors", "Motion, HR, Clock"): (0.648, 0.647, 0.721, 0.243),
    ("Random Forest", "Motion only"): (0.397, 0.441, 0.702, 0.075), ("Random Forest", "HR only"): (0.434, 0.434, 0.676, 0.165),
    ("Random Forest", "Motion, HR"): (0.615, 0.615, 0.695, 0.293), ("Random Forest", "Motion, HR, Clock"): (0.638, 0.638, 0.686, 0.302),
    ("Neural Net", "Motion only"): (0.394, 0.498, 0.713, 0.084), ("Neural Net", "HR only"): (0.454, 0.454, 0.698, 0.040),
    ("Neural Net", "Motion, HR"): (0.622, 0.622, 0.723, 0.256), ("Neural Net", "Motion, HR, Clock"): (0.651, 0.650, 0.723, 0.277),
}
CLASSIFIERS_RU = {"Logistic Regression": "Логистическая регрессия", "k-Nearest Neighbors": "k ближайших соседей",
                  "Random Forest": "Случайный лес", "Neural Net": "Нейросеть (MLP)"}


def bolded(df, cols, group=None, lower_is_better=(), drop=()):
    """Строки таблицы с выделенным жирным лучшим значением в каждой колонке (внутри группы, если задана).
    Колонки из drop участвуют в группировке, но в таблицу не попадают."""
    df = df.reset_index(drop=True)
    best = {}
    groups = df.groupby(group).groups.values() if group else [df.index]
    for idx in groups:
        for c in cols:
            vals = pd.to_numeric(df.loc[idx, c], errors="coerce")
            if vals.notna().any():
                target = vals.min() if c in lower_is_better else vals.max()
                for i in vals.index[vals == target]:
                    best[(i, c)] = True
    rows = []
    for i, row in df.iterrows():
        rows.append([f"**{fmt(v)}**" if best.get((i, c)) and fmt(v) else fmt(v)
                     for c, v in row.items() if c not in drop])
    return rows


def section_paper_metrics(adir):
    a = adir / "paper_metrics"
    out = ["## Все метрики статьи для четырёх моделей",
           "Протокол: **leave-one-subject-out** (31 фолд): каждый человек предсказывается моделью, обученной на "
           "остальных 30. Модели и гиперпараметры — из кода статьи, фиксированные значения из её сеток "
           "(`repro/09_paper_metrics.py`). В статье — Monte Carlo 70/30 (50 и 20 разбиений) с подбором "
           "гиперпараметров на каждом, поэтому числа близки, но не обязаны совпадать до третьего знака. "
           "**Жирным** — лучшее значение среди наборов признаков. Таблицы 7–9 и рис. 8–9 (MESA) не воспроизводимы: "
           "данные MESA выдаются только по заявке NSRR."]
    t1 = read_csv(a / "table1_summary.csv")
    if t1 is None:
        return out + ["_не посчитано: запустите `python repro/run_all.py` (или `python repro/09_paper_metrics.py`)_"]
    read_text = lambda name, default: (a / name).read_text(encoding="utf-8").strip() if (a / name).exists() else default
    models_done = read_text("models_included.txt", "").splitlines()
    ps_model = read_text("per_subject_model.txt", "Neural Net")
    ps_model_ru = CLASSIFIERS_RU.get(ps_model, ps_model).lower()
    if models_done and "Neural Net" not in models_done:
        out.append("> **Нейросеть ещё не обучена** — в таблицах только: " +
                   ", ".join(CLASSIFIERS_RU[m] for m in models_done) +
                   f". Рисунки 5–7 (в статье — по нейросети) временно построены по модели «{CLASSIFIERS_RU[ps_model]}»; "
                   "фиксированные пороги статьи θW = 0.3 и θREM = 0.35 подобраны под нейросеть без весов классов, "
                   "поэтому для этой модели сдвиги в Bland–Altman завышены. Дообучить: "
                   "`python repro/09_paper_metrics.py --models \"Neural Net\"` — остальное возьмётся из сохранённого.")
    source = (a / "table1_source.txt").read_text(encoding="utf-8") if (a / "table1_source.txt").exists() else ""
    out += [f"### Таблица 1. Сон по PSG (31 человек; источник — {source})",
            df_table(t1, None, {"показатель": "показатель"}),
            "TIB, TST и время в стадиях у нас меньше, а разброс больше: в открытых данных у 5 человек разметка ночи "
            "обрезана до 3.5–4.7 ч (у часов села батарея), а статья считала Таблицу 1 по полной PSG. Показатели, "
            "не зависящие от длины ночи (SE, SOL, WASO), совпадают. Возраста в открытых данных нет "
            "(в статье 29.4 ± 8.5 года, 21 женщина)."]

    sw = read_csv(a / "sleep_wake_all_models.csv")
    tc = read_csv(a / "three_class_all_models.csv")
    if sw is not None:
        d = sw[sw.sensitivity_target.round(2) == 0.9]
        rows = []
        for _, r in d.iterrows():
            p = PAPER_SW.get((r.classifier, r.feature_set), (np.nan,) * 3)
            rows.append([CLASSIFIERS_RU[r.classifier], r.feature_set, r.auc, p[0], r.wake_correct, p[1], r.kappa, p[2],
                         r.f1_wake, r.f1_macro])
        frame = pd.DataFrame(rows, columns=["модель", "признаки", "AUC", "AUC статья", "wake найден", "wake статья",
                                            "κ", "κ статья", "F1 wake", "macro F1"])
        out += ["### Сводка: наши значения рядом со статьёй (сон найден = 0.90)",
                md_table(bolded(frame, ["AUC", "wake найден", "κ", "F1 wake", "macro F1"], group="модель"), list(frame.columns))]

        out.append("### Таблицы 2–5. Сон/бодрствование: все пороги и все метрики")
        for clf in sw.classifier.unique():
            d = sw[sw.classifier == clf]
            frame = pd.DataFrame({
                "признаки": d.feature_set, "сон найден": d.sensitivity_target, "accuracy": d.accuracy,
                "wake найден": d.wake_correct, "κ": d.kappa, "AUC": d.auc, "AP wake": d.ap_wake,
                "PPV wake": d.ppv_wake, "PPV сон": d.ppv_sleep, "F1 wake": d.f1_wake, "F1 сон": d.f1_sleep,
                "macro F1": d.f1_macro})
            frame.loc[frame.duplicated(["признаки"]), ["AUC", "AP wake"]] = np.nan  # AUC/AP не зависят от порога
            frame["признаки"] = frame["признаки"].where(~frame.duplicated(["признаки"]), "")
            cols = ["accuracy", "wake найден", "κ", "AUC", "AP wake", "PPV wake", "F1 wake", "macro F1"]
            out += [f"**{CLASSIFIERS_RU[clf]}**",
                    md_table(bolded(frame.assign(_g=d.sensitivity_target.values), cols, group="_g", drop=["_g"]),
                             list(frame.columns))]
    if tc is not None:
        rows = []
        for _, r in tc.iterrows():
            p = PAPER_T6.get((r.classifier, r.feature_set), (np.nan,) * 4)
            rows.append([CLASSIFIERS_RU[r.classifier], r.feature_set, r.nrem_correct, p[0], r.rem_correct, p[1],
                         r.best_accuracy, p[2], r.kappa, p[3], r.f1_macro_best, r.f1_macro_balanced, r.f1_rem_balanced])
        frame = pd.DataFrame(rows, columns=["модель", "признаки", "NREM найден", "NREM статья", "REM найден",
                                            "REM статья", "лучшая accuracy", "accuracy статья", "κ", "κ статья",
                                            "macro F1 (лучшая acc.)", "macro F1 (NREM=REM)", "F1 REM (NREM=REM)"])
        out += ["### Таблица 6. Wake / NREM / REM (wake найден = 0.6)",
                md_table(bolded(frame, ["NREM найден", "REM найден", "лучшая accuracy", "κ", "macro F1 (лучшая acc.)",
                                        "macro F1 (NREM=REM)", "F1 REM (NREM=REM)"], group="модель"), list(frame.columns)),
                "«NREM=REM» — порог, при котором доли найденных NREM и REM уравнены (как сплошные линии на рис. 4); "
                "«лучшая accuracy» — точка, где статья берёт κ. Именно там видно, как accuracy ≈ 0.70 достигается почти "
                "без REM."]
        fixed = tc[["classifier", "feature_set", "f1_wake_fixed", "f1_nrem_fixed", "f1_rem_fixed", "f1_macro_fixed"]].copy()
        fixed["classifier"] = fixed.classifier.map(CLASSIFIERS_RU)
        fixed.columns = ["модель", "признаки", "F1 wake", "F1 NREM", "F1 REM", "macro F1"]
        out += ["**F1 по стадиям при фиксированных порогах статьи** (θW = 0.3, θREM = 0.35, как для рис. 5)",
                md_table(bolded(fixed, ["F1 wake", "F1 NREM", "F1 REM", "macro F1"], group="модель"), list(fixed.columns))]

    ba = read_csv(a / "bland_altman_stats.csv")
    if ba is not None:
        names = {"TST": "TST, мин", "SOL": "SOL, мин", "WASO": "WASO, мин", "SE": "SE (доля)", "REM": "REM, мин",
                 "NREM": "NREM, мин"}
        ba = ba.assign(metric=ba.metric.map(names))
        ba["границы согласия"] = ba.apply(lambda r: f"[{r.loa_low:.2f}; {r.loa_high:.2f}]", axis=1)
        frame = ba[["metric", "feature_set", "bias", "sd", "границы согласия"]].rename(
            columns={"metric": "показатель", "feature_set": "признаки", "bias": "сдвиг (PSG − модель)", "sd": "SD"})
        frame["показатель"] = frame["показатель"].where(~frame.duplicated(["показатель"]), "")
        out += [f"### Рисунок 5. Bland–Altman: показатели сна ({ps_model_ru}, θW = 0.3, θREM = 0.35)",
                "Положительный сдвиг — модель **недооценивает** показатель. Чем уже границы согласия, тем надёжнее "
                "оценка для отдельного человека.",
                df_table(frame)]

    p6 = read_csv(a / "per_subject_theta.csv")
    p7 = read_csv(a / "per_subject_tpr.csv")
    if p6 is not None:
        desc = p6.drop(columns=["subject"]).describe().T[["min", "25%", "50%", "75%", "max"]].reset_index()
        desc.columns = ["метрика", "мин", "25%", "медиана", "75%", "макс"]
        desc["метрика"] = desc["метрика"].map({"accuracy": "accuracy", "wake_correct": "wake найден",
                                               "sleep_correct": "сон найден", "kappa": "κ", "f1_wake": "F1 wake"})
        out += [f"### Рисунок 6. Качество по людям ({ps_model_ru}, Motion, HR, Clock, P(wake) ≥ 0.3 → wake)",
                df_table(desc)]
    if p7 is not None:
        g = p7.groupby("tpr")[["accuracy", "wake_correct"]].median().reset_index()
        g.columns = ["сон найден (порог под человека)", "accuracy, медиана", "wake найден, медиана"]
        out += ["### Рисунок 7. Порог под каждого человека", df_table(g)]
    return out


GALLERY = [  # (папка относительно результатов, файл, подпись) — в порядке показа
    ("paper_metrics", "fig_summary_models.png", "Все модели статьи и наборы признаков: AUC, κ, F1 wake"),
    ("paper_metrics", "fig2_roc_sleep_wake.png", "Рис. 2: ROC сон/бодрствование, 4 модели"),
    ("paper_metrics", "fig3_pr_wake.png", "Рис. 3: precision–recall для wake"),
    ("paper_metrics", "fig4_three_class.png", "Рис. 4: wake / NREM / REM"),
    ("paper_metrics", "fig5_bland_altman.png", "Рис. 5: Bland–Altman показателей сна"),
    ("paper_metrics", "fig6_per_subject_theta.png", "Рис. 6: качество по людям при фиксированном пороге"),
    ("paper_metrics", "fig7_per_subject_tpr.png", "Рис. 7: качество по людям при пороге под человека"),
    ("f1", "f1_vs_accuracy.png", "Accuracy против F1 (с моделью «всегда сон»)"),
    ("f1", "f1_threshold.png", "F1 бодрствования в зависимости от порога"),
    ("f1", "f1_three_class_and_subjects.png", "F1 по стадиям и по людям"),
    ("importance", "A_subsets_shapley.png", "Важность признаков: подмножества и значения Шепли"),
    ("importance", "B_model_weights.png", "Веса моделей"),
    ("importance", "C_permutation.png", "Permutation importance"),
    ("importance", "D_shap_beeswarm_lgbm.png", "SHAP (LightGBM)"),
    ("importance", "D_shap_three_class.png", "SHAP по стадиям"),
    ("importance", "D_shap_over_night_46343.png", "Что «говорит» каждый признак по ходу ночи"),
    ("logreg", "logreg_overview.png", "Логистическая регрессия: сигмоида, калибровка, регуляризация, бутстреп"),
    ("logreg", "logreg_linearity_check.png", "Проверка линейности по log-odds"),
    ("improve", "roc_paper_vs_improved.png", "Модель статьи против улучшенных"),
    ("improve", "shap_lgbm_ext_sw.png", "SHAP расширенной модели"),
    ("time_of_night", "quality_by_hour.png", "Качество по часам ночи"),
    ("dataset", "coverage.png", "Покрытие сигналов по испытуемым"),
    ("dataset", "stage_distribution.png", "Распределение стадий по испытуемым"),
    ("dataset", "subject_46343_night.png", "Пример ночи: ускорение, ENMO, пульс, гипнограмма"),
]


def section_gallery(adir, fig_dirs, out_path):
    import os
    out = ["## Графики",
           "Ключевые рисунки прямо здесь; полный список файлов — ниже."]
    shown = set()
    for folder, name, caption in GALLERY:
        p = adir / folder / name
        if p.exists():
            rel = Path(os.path.relpath(p, out_path.parent)).as_posix().replace(" ", "%20")
            out.append(f"**{caption}**\n\n![{caption}]({rel})")
            shown.add(p.resolve())
    paper_figs = [d / n for d in fig_dirs for n in ("figure_50_sw_roc.png", "figure_50_sw_pr.png",
                                                    "figure_20_three_class_roc.png")]
    for p in paper_figs:
        p2 = p if p.exists() else p.parent / "figures_paper" / p.name
        if p2.exists() and p2.resolve() not in shown:
            rel = Path(os.path.relpath(p2, out_path.parent)).as_posix().replace(" ", "%20")
            out.append(f"**Оригинальный код статьи: {p2.name}**\n\n![{p2.name}]({rel})")
            shown.add(p2.resolve())
    out.append("### Все остальные рисунки")
    for d in fig_dirs:
        pngs = sorted(d.rglob("*.png")) if d.exists() else []
        pngs = [p for p in pngs if not p.name.startswith("data_validation_") and p.resolve() not in shown]
        for p in pngs:
            rel = Path(os.path.relpath(p, out_path.parent)).as_posix()
            out.append(f"- [{p.relative_to(d).as_posix()}]({rel.replace(' ', '%20')})")
    out.append("Рисунок 1 статьи (пример ночи) для каждого испытуемого — файлы `data_validation_<id>.png`.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Дамп всех результатов в Markdown")
    parser.add_argument("--snapshot", action="store_true", help="брать результаты из results/ (снимок в репозитории)")
    parser.add_argument("--out", default=None, help="куда записать (по умолчанию RESULTS.md рядом с результатами)")
    args = parser.parse_args()
    if args.snapshot:
        tables_dir, analysis_dir = ROOT / "results" / "tables", ROOT / "results"
        fig_dirs = [ROOT / "results"]
        out_path = Path(args.out) if args.out else ROOT / "results" / "RESULTS.md"
    else:
        tables_dir, analysis_dir = ROOT / "outputs" / "logs", ROOT / "outputs" / "repro"
        fig_dirs = [ROOT / "outputs" / "figures", ROOT / "outputs" / "repro"]
        out_path = Path(args.out) if args.out else ROOT / "outputs" / "RESULTS.md"
    out_path = out_path if out_path.is_absolute() else ROOT / out_path

    sections = [section_paper_metrics(analysis_dir), section_paper(tables_dir), section_dataset(analysis_dir),
                section_importance(analysis_dir), section_logreg(analysis_dir), section_improve(analysis_dir),
                section_time(analysis_dir), section_f1(analysis_dir), section_gallery(analysis_dir, fig_dirs, out_path)]
    # единая нумерация разделов «## N. …» и содержание
    titles = []
    for section in sections:
        title = re.sub(r"^## (\d+\.\s*)?", "", section[0])
        titles.append(title)
        section[0] = f"## {len(titles)}. {title}"
    parts = ["# Результаты проекта: предсказание стадий сна по Apple Watch",
             "Сгенерировано `repro/08_results_markdown.py` из готовых результатов (ничего не пересчитывается). "
             "Подробнее о модели и метриках — `docs/LOGREG_RU.md` и `docs/F1_RU.md`.",
             "**Содержание:** " + " · ".join(f"{i}. {t}" for i, t in enumerate(titles, 1)),
             "**Метрики, которые встречаются ниже**",
             md_table(GLOSSARY, ["метрика", "что значит"])]
    for section in sections:
        parts += section
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"записано: {out_path}  ({out_path.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
