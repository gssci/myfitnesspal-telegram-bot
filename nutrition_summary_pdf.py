"""Create a compact seven-day A4 nutrition summary from a MyFitnessPal CSV export."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Mapping

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REQUIRED_COLUMNS = ("Date", "Meal", "Calories", "Fat (g)", "Carbohydrates (g)", "Protein (g)")
HEALTH_COLUMNS = {
    "steps": "Steps (sum)",
    "energy": "Total Energy (sum)",
    "sleep": "Sleep (sum)",
    "weight": "Weight (Most Recent) (most recent)",
    "body_fat": "Body Fat % (most recent)",
}
HEALTH_METRICS = (
    ("steps", "#2A9D8F"),
    ("energy", "#E76F51"),
    ("sleep", "#7B61B8"),
    ("weight", "#4C78A8"),
    ("body_fat", "#C9698C"),
)
MEAL_ORDER = ("Breakfast", "Lunch", "Dinner", "Snacks")
MACROS = (
    ("protein", 4, "#507AA6"),
    ("carbohydrates", 4, "#D9A441"),
    ("fat", 9, "#C9695B"),
)
Language = Literal["english", "italian"]

TEXT = {
    "english": {
        "title": "Weekly nutrition summary",
        "logged_days": "Logged days",
        "one_day_with_data": "day with data",
        "many_days_with_data": "days with data",
        "one_excluded_day": "excluded day",
        "many_excluded_days": "excluded days",
        "no_data": "No logged data",
        "no_nutrition_data": "No nutrition data",
        "excluded_day": "Excluded day",
        "health": "HEALTH",
        "no_health_data": "No health data",
        "health_labels": {"steps": "Steps", "energy": "Energy", "sleep": "Sleep", "weight": "Weight", "body_fat": "Body fat"},
        "calories_consumed": "kcal consumed",
        "fiber": "Fibre",
        "sodium": "Sodium",
        "meals": "MEALS",
        "extra_meals": "+ {count} additional meal(s)",
        "macro_footer": "Macro split shows estimated energy contribution (protein 4 kcal/g, carbohydrate 4 kcal/g, fat 9 kcal/g). ",
        "source_footer": "Meal and nutrient values are aggregated from the supplied CSV; label calories can differ from macro energy because of rounding and other nutrients.",
        "metadata_title": "Weekly nutrition summary",
        "macro_names": {"protein": "Protein", "carbohydrates": "Carbs", "fat": "Fat"},
        "meal_names": {},
        "weekdays": ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
        "months": ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
    },
    "italian": {
        "title": "Riepilogo nutrizionale settimanale",
        "logged_days": "Giorni registrati",
        "one_day_with_data": "giorno con dati",
        "many_days_with_data": "giorni con dati",
        "one_excluded_day": "giorno escluso",
        "many_excluded_days": "giorni esclusi",
        "no_data": "Nessun dato registrato",
        "no_nutrition_data": "Nessun dato nutrizionale",
        "excluded_day": "Giorno escluso",
        "health": "SALUTE",
        "no_health_data": "Dati salute non disponibili",
        "health_labels": {"steps": "Passi", "energy": "Consumo", "sleep": "Sonno", "weight": "Peso", "body_fat": "Grasso"},
        "calories_consumed": "kcal consumate",
        "fiber": "Fibre",
        "sodium": "Sodio",
        "meals": "PASTI",
        "extra_meals": "+ altri {count} pasti",
        "macro_footer": "La ripartizione dei macronutrienti mostra il contributo energetico stimato (proteine 4 kcal/g, carboidrati 4 kcal/g, grassi 9 kcal/g). ",
        "source_footer": "I valori dei pasti e dei nutrienti sono aggregati dal CSV fornito; le kcal in etichetta possono differire dall'energia dei macronutrienti per arrotondamenti e altri nutrienti.",
        "metadata_title": "Riepilogo nutrizionale settimanale",
        "macro_names": {"protein": "Proteine", "carbohydrates": "Carboidrati", "fat": "Grassi"},
        "meal_names": {"breakfast": "Colazione", "lunch": "Pranzo", "dinner": "Cena", "snacks": "Spuntini"},
        "weekdays": ("lun", "mar", "mer", "gio", "ven", "sab", "dom"),
        "months": ("gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"),
    },
}


@dataclass
class MealTotal:
    name: str
    calories: float = 0.0
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0


@dataclass
class DayTotal:
    day: date
    has_nutrition: bool = True
    calories: float = 0.0
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sodium: float = 0.0
    meals: OrderedDict[str, MealTotal] = field(default_factory=OrderedDict)


@dataclass
class HealthTotal:
    day: date
    steps: float | None = None
    energy: float | None = None
    sleep: float | None = None
    weight: float | None = None
    body_fat: float | None = None


def _number(value: str | None) -> float:
    """Parse an export value, treating blank cells as zero."""
    if value is None or not value.strip():
        return 0.0
    try:
        return float(value.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"Expected a number, got {value!r}") from error


def _optional_number(value: str | None) -> float | None:
    return None if value is None or not value.strip() else _number(value)


def _column(row: dict[str, str], name: str) -> str | None:
    """Find a column while accepting incidental whitespace in CSV headers."""
    for header, value in row.items():
        if header and header.strip() == name:
            return value
    return None


def _meal_sort_key(name: str) -> tuple[int, str]:
    normalized = name.strip().casefold()
    for index, known_meal in enumerate(MEAL_ORDER):
        if normalized == known_meal.casefold():
            return index, normalized
    return len(MEAL_ORDER), normalized


def read_days(csv_path: Path) -> list[DayTotal]:
    """Aggregate all rows in a MyFitnessPal nutrition summary by day and meal."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("The CSV has no header row.")
        headers = {header.strip() for header in reader.fieldnames if header}
        missing = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))

        days: dict[date, DayTotal] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                day = datetime.strptime((_column(row, "Date") or "").strip(), "%Y-%m-%d").date()
            except ValueError as error:
                raise ValueError(f"Row {line_number}: Date must use YYYY-MM-DD.") from error

            meal_name = (_column(row, "Meal") or "Unspecified").strip() or "Unspecified"
            day_total = days.setdefault(day, DayTotal(day=day))
            meal = day_total.meals.setdefault(meal_name, MealTotal(name=meal_name))
            values = {
                "calories": _number(_column(row, "Calories")),
                "fat": _number(_column(row, "Fat (g)")),
                "carbohydrates": _number(_column(row, "Carbohydrates (g)")),
                "protein": _number(_column(row, "Protein (g)")),
                "fiber": _number(_column(row, "Fiber")),
                "sodium": _number(_column(row, "Sodium (mg)")),
            }
            for name, amount in values.items():
                setattr(day_total, name, getattr(day_total, name) + amount)
            for name in ("calories", "protein", "carbohydrates", "fat"):
                setattr(meal, name, getattr(meal, name) + values[name])

    for day_total in days.values():
        day_total.meals = OrderedDict(sorted(day_total.meals.items(), key=lambda item: _meal_sort_key(item[0])))
    return [days[day] for day in sorted(days)]


def read_health_days(csv_path: Path) -> dict[date, HealthTotal]:
    """Read the daily Apple Health-style export used to complement the nutrition report."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("The health CSV has no header row.")
        headers = {header.strip() for header in reader.fieldnames if header}
        required = ("Date", *HEALTH_COLUMNS.values())
        missing = [column for column in required if column not in headers]
        if missing:
            raise ValueError("Health CSV is missing required columns: " + ", ".join(missing))

        days: dict[date, HealthTotal] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                day = datetime.strptime((_column(row, "Date") or "").strip(), "%Y-%m-%d").date()
            except ValueError as error:
                raise ValueError(f"Health CSV row {line_number}: Date must use YYYY-MM-DD.") from error
            health = days.setdefault(day, HealthTotal(day=day))
            for attribute, column in HEALTH_COLUMNS.items():
                value = _optional_number(_column(row, column))
                if value is not None:
                    setattr(health, attribute, value)
    return days


def select_report_days(days: list[DayTotal], start_date: date | None) -> list[DayTotal]:
    """Return seven calendar days, retaining placeholders for missing nutrition days."""
    if not days:
        return []
    by_day = {day.day: day for day in days}
    first_day = start_date or (max(by_day) - timedelta(days=6))
    return [
        by_day.get(slot, DayTotal(day=slot, has_nutrition=False))
        for slot in (first_day + timedelta(days=offset) for offset in range(7))
    ]


def _macro_energy(day: DayTotal) -> list[float]:
    return [getattr(day, attribute) * calories_per_gram for attribute, calories_per_gram, _ in MACROS]


def _day_month(day: date, language: Language) -> str:
    """Format a date without system locale or platform-specific strftime directives."""
    return f"{day.day} {TEXT[language]['months'][day.month - 1]}"


def _format_day(day: date, language: Language) -> str:
    return _day_month(day, language) + f" {day.year}"


def _display_meal_name(name: str, language: Language) -> str:
    return TEXT[language]["meal_names"].get(name.casefold(), name)


def _days_with_data(count: int, language: Language) -> str:
    key = "one_day_with_data" if count == 1 else "many_days_with_data"
    return f"{count} {TEXT[language][key]}"


def _excluded_days(count: int, language: Language) -> str:
    key = "one_excluded_day" if count == 1 else "many_excluded_days"
    return f"{count} {TEXT[language][key]}"


def _format_health_value(metric: str, value: float | None) -> str:
    if value is None:
        return "—"
    if metric == "steps":
        return f"{value:.0f}"
    if metric == "energy":
        return f"{value:,.0f} kcal"
    if metric == "sleep":
        return f"{value:.1f} h"
    if metric == "weight":
        return f"{value:.1f} kg"
    if metric == "body_fat":
        return f"{value:.1f}%"
    raise ValueError(f"Unknown health metric: {metric}")


def _add_macro_legend(axis, day: DayTotal, language: Language) -> None:
    energies = _macro_energy(day)
    total_energy = sum(energies)
    for index, ((attribute, _, color), energy) in enumerate(zip(MACROS, energies)):
        y = 0.746 - index * 0.047
        grams = getattr(day, attribute)
        percentage = 0 if total_energy == 0 else energy / total_energy * 100
        axis.add_patch(plt.Rectangle((0.535, y - 0.009), 0.020, 0.018, color=color, transform=axis.transAxes, clip_on=False))
        label = TEXT[language]["macro_names"][attribute]
        axis.text(0.575, y, f"{label[0]}  {grams:.0f} g  ·  {percentage:.0f}%", transform=axis.transAxes,
                  va="center", fontsize=5.5, color="#263238")


def _draw_health_icon(axis, x: float, y: float, metric: str, color: str) -> None:
    """Draw screen-space markers so circular icon backgrounds cannot be axis-distorted."""
    axis.scatter([x], [y], s=155, marker="o", c=[color], linewidths=0, transform=axis.transAxes, clip_on=False, zorder=2)
    if metric == "steps":
        axis.scatter([x - 0.005, x + 0.006], [y + 0.003, y - 0.006], s=[12, 12], marker="o", c="white", linewidths=0, transform=axis.transAxes, clip_on=False, zorder=3)
    elif metric == "energy":
        axis.scatter([x], [y], s=38, marker="^", c="white", linewidths=0, transform=axis.transAxes, clip_on=False, zorder=3)
    elif metric == "sleep":
        axis.scatter([x - 0.002], [y], s=45, marker="o", c="white", linewidths=0, transform=axis.transAxes, clip_on=False, zorder=3)
        axis.scatter([x + 0.005], [y + 0.004], s=45, marker="o", c=[color], linewidths=0, transform=axis.transAxes, clip_on=False, zorder=4)
    elif metric == "weight":
        axis.scatter([x], [y], s=42, marker="s", c="white", linewidths=0, transform=axis.transAxes, clip_on=False, zorder=3)
        axis.scatter([x], [y + 0.004], s=7, marker="o", c=[color], linewidths=0, transform=axis.transAxes, clip_on=False, zorder=4)
    else:
        axis.text(x, y - 0.001, "%", ha="center", va="center", fontsize=4.5, fontweight="bold", color="white", transform=axis.transAxes, zorder=3)


def _draw_health_metric(axis, y: float, metric: str, value: float | None, language: Language) -> None:
    color = dict(HEALTH_METRICS)[metric]
    _draw_health_icon(axis, 0.075, y, metric, color)
    axis.text(0.125, y, TEXT[language]["health_labels"][metric].upper(), ha="left", va="center", fontsize=4.6,
              fontweight="bold", color="#718096", transform=axis.transAxes)
    axis.text(0.925, y, _format_health_value(metric, value), ha="right", va="center", fontsize=5.8,
              fontweight="bold", color="#263238", transform=axis.transAxes)


def _draw_health_section(axis, health: HealthTotal | None, language: Language) -> None:
    axis.plot([0.07, 0.93], [0.295, 0.295], color="#D8DEE4", linewidth=0.6, transform=axis.transAxes)
    axis.text(0.075, 0.273, TEXT[language]["health"], fontsize=5.0, fontweight="bold", color="#718096", transform=axis.transAxes)
    if health is None:
        axis.text(0.5, 0.170, TEXT[language]["no_health_data"], ha="center", va="center", fontsize=5.0, color="#718096", transform=axis.transAxes)
        return
    for index, metric in enumerate(("steps", "energy", "sleep", "weight", "body_fat")):
        y = 0.234 - index * 0.042
        _draw_health_metric(axis, y, metric, getattr(health, metric), language)
        if index < 4:
            axis.plot([0.125, 0.925], [y - 0.021, y - 0.021], color="#E7EBEF", linewidth=0.45, transform=axis.transAxes)


def _draw_day(
    axis,
    day: DayTotal | None,
    language: Language,
    excluded: bool = False,
    health: HealthTotal | None = None,
) -> None:
    axis.set_axis_off()
    background = "#EEF1F4" if excluded else "#FAFBFC"
    border = "#CAD1D8" if excluded else "#D8DEE4"
    axis.add_patch(plt.Rectangle((0.02, 0.01), 0.96, 0.98, facecolor=background, edgecolor=border,
                                 linewidth=0.7, transform=axis.transAxes, zorder=-2))
    if day is None:
        axis.text(0.5, 0.53, TEXT[language]["no_data"], ha="center", va="center", fontsize=7, color="#718096",
                  transform=axis.transAxes)
        return

    axis.text(0.5, 0.95, TEXT[language]["weekdays"][day.day.weekday()], ha="center", va="top", fontsize=8.3, fontweight="bold",
              color="#172B4D", transform=axis.transAxes)
    axis.text(0.5, 0.915, _day_month(day.day, language), ha="center", va="top", fontsize=6.3, color="#52616B",
              transform=axis.transAxes)
    if excluded:
        axis.text(0.5, 0.53, TEXT[language]["excluded_day"], ha="center", va="center", fontsize=7.0,
                  fontweight="bold", color="#718096", transform=axis.transAxes)
        return
    if not day.has_nutrition:
        axis.text(0.5, 0.53, TEXT[language]["no_nutrition_data"], ha="center", va="center", fontsize=6.5,
                  color="#718096", transform=axis.transAxes)
        _draw_health_section(axis, health, language)
        return
    axis.text(0.5, 0.862, f"{day.calories:,.0f}", ha="center", va="top", fontsize=11.2, fontweight="bold",
              color="#172B4D", transform=axis.transAxes)
    axis.text(0.5, 0.823, TEXT[language]["calories_consumed"], ha="center", va="top", fontsize=5.8, color="#52616B",
              transform=axis.transAxes)

    pie_axis = axis.inset_axes([0.075, 0.625, 0.405, 0.170])
    energies = _macro_energy(day)
    if sum(energies):
        pie_axis.pie(energies, colors=[macro[2] for macro in MACROS], startangle=90,
                     wedgeprops={"linewidth": 0.8, "edgecolor": "white"})
    else:
        pie_axis.pie([1], colors=["#E3E8ED"], wedgeprops={"linewidth": 0})
    pie_axis.set_aspect("equal")
    pie_axis.set_axis_off()
    _add_macro_legend(axis, day, language)

    extras = f"{TEXT[language]['fiber']} {day.fiber:.0f} g"
    if day.sodium:
        extras += f"  ·  {TEXT[language]['sodium']} {day.sodium:,.0f} mg"
    axis.text(0.5, 0.605, extras, ha="center", va="center", fontsize=5.0, color="#52616B",
              transform=axis.transAxes)
    axis.plot([0.07, 0.93], [0.582, 0.582], color="#D8DEE4", linewidth=0.6, transform=axis.transAxes)
    axis.text(0.075, 0.560, TEXT[language]["meals"], fontsize=5.0, fontweight="bold", color="#718096", transform=axis.transAxes)

    meals = list(day.meals.values())
    for index, meal in enumerate(meals[:5]):
        y = 0.525 - index * 0.056
        axis.text(0.075, y, _display_meal_name(meal.name, language), fontsize=5.7, fontweight="bold", color="#263238", transform=axis.transAxes)
        axis.text(0.925, y, f"{meal.calories:.0f} kcal", ha="right", fontsize=5.7, color="#263238",
                  transform=axis.transAxes)
        axis.text(0.075, y - 0.023, f"P {meal.protein:.0f}  C {meal.carbohydrates:.0f}  F {meal.fat:.0f} g",
                  fontsize=4.9, color="#52616B", transform=axis.transAxes)
    if len(meals) > 5:
        axis.text(0.075, 0.02, TEXT[language]["extra_meals"].format(count=len(meals) - 5), fontsize=4.8, color="#718096",
                  transform=axis.transAxes)
    _draw_health_section(axis, health, language)


def create_pdf(
    days: Iterable[DayTotal],
    output_path: Path,
    language: Language = "english",
    excluded_dates: Iterable[date] = (),
    health_days: Mapping[date, HealthTotal] | None = None,
) -> None:
    """Render exactly one landscape A4 PDF page with seven day columns."""
    report_days = list(days)
    padded_days: list[DayTotal | None] = (report_days + [None] * 7)[:7]
    excluded = set(excluded_dates)
    excluded_in_report = {day.day for day in report_days if day.day in excluded}
    if not report_days:
        raise ValueError("No data matched the selected date range.")

    figure = plt.figure(figsize=(11.6929, 8.2677), facecolor="white")  # A4 landscape in inches
    grid = figure.add_gridspec(1, 7, left=0.026, right=0.974, top=0.965, bottom=0.055, wspace=0.075)
    health_by_day = health_days or {}

    for index, day in enumerate(padded_days):
        _draw_day(
            figure.add_subplot(grid[0, index]),
            day,
            language,
            day is not None and day.day in excluded_in_report,
            health_by_day.get(day.day) if day is not None else None,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path, metadata={"Title": TEXT[language]["metadata_title"]}) as pdf:
        pdf.savefig(figure, dpi=220)
    plt.close(figure)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="MyFitnessPal nutrition-summary CSV export")
    parser.add_argument("-o", "--output", type=Path, default=Path("nutrition-summary.pdf"), help="PDF output path")
    parser.add_argument("--start-date", type=_parse_date, help="First calendar date to include (YYYY-MM-DD); includes the next six calendar days")
    parser.add_argument("--exclude-date", action="append", type=_parse_date, default=[], metavar="YYYY-MM-DD",
                        help="Exclude a date's values while retaining it as a grey report column; repeat for multiple dates")
    parser.add_argument("--health-csv", type=Path,
                        help="Daily health-export CSV with steps, total energy, sleep, weight, and body-fat data")
    parser.add_argument("--language", choices=("english", "italian"), default="english", help="Report language (default: english)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.csv_file.is_file():
        raise SystemExit(f"CSV file not found: {args.csv_file}")
    if args.health_csv and not args.health_csv.is_file():
        raise SystemExit(f"Health CSV file not found: {args.health_csv}")
    try:
        report_days = select_report_days(read_days(args.csv_file), args.start_date)
        health_days = read_health_days(args.health_csv) if args.health_csv else None
        create_pdf(report_days, args.output, args.language, args.exclude_date, health_days)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Unable to create report: {error}") from error
    excluded_count = sum(day.has_nutrition and day.day in args.exclude_date for day in report_days)
    nutrition_count = sum(day.has_nutrition for day in report_days)
    print(f"Created {args.output.resolve()} ({nutrition_count - excluded_count} included, {excluded_count} excluded day(s)).")


if __name__ == "__main__":
    main()
