import tempfile
import unittest
from datetime import date
from pathlib import Path

from nutrition_summary_pdf import _day_month, _days_with_data, _display_meal_name, _excluded_days, _format_health_value, create_pdf, read_days, read_health_days, select_report_days


class NutritionSummaryTests(unittest.TestCase):
    def _write_csv(self, directory: Path) -> Path:
        path = directory / "input.csv"
        path.write_text(
            "Date,Meal,Calories,Fat (g),Carbohydrates (g),Protein (g),Fiber,Sodium (mg)\n"
            "2026-07-14,Breakfast,300,10,20,30,4,200\n"
            "2026-07-14,Lunch,500,20,60,40,8,300\n"
            "2026-07-15,Dinner,600,25,50,45,6,400\n",
            encoding="utf-8",
        )
        return path

    def _write_health_csv(self, directory: Path) -> Path:
        path = directory / "health.csv"
        path.write_text(
            "Date,Steps (sum),Total Energy (sum),Sleep (sum),Weight (Most Recent) (most recent),Body Fat % (most recent)\n"
            "2026-07-14,3400,2762.9,7.2,93.2,22.0\n"
            "2026-07-15,14201,3490.9,,,21.9\n",
            encoding="utf-8",
        )
        return path

    def test_aggregates_meals_and_days(self):
        with tempfile.TemporaryDirectory() as directory:
            days = read_days(self._write_csv(Path(directory)))
        self.assertEqual(len(days), 2)
        self.assertEqual(days[0].calories, 800)
        self.assertEqual(days[0].protein, 70)
        self.assertEqual(days[0].fiber, 12)
        self.assertEqual(list(days[0].meals), ["Breakfast", "Lunch"])

    def test_start_date_and_pdf_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            days = read_days(self._write_csv(directory_path))
            health_days = read_health_days(self._write_health_csv(directory_path))
            selected = select_report_days(days, date(2026, 7, 15))
            output = directory_path / "report.pdf"
            create_pdf(selected, output, excluded_dates=[date(2026, 7, 15)], health_days=health_days)
            self.assertEqual([day.day for day in selected], [date(2026, 7, 15), date(2026, 7, 16), date(2026, 7, 17), date(2026, 7, 18), date(2026, 7, 19), date(2026, 7, 20), date(2026, 7, 21)])
            self.assertFalse(selected[1].has_nutrition)
            self.assertEqual(health_days[date(2026, 7, 14)].steps, 3400)
            self.assertIsNone(health_days[date(2026, 7, 15)].sleep)
            self.assertTrue(output.read_bytes().startswith(b"%PDF"))

    def test_italian_labels_are_localized_without_system_locale(self):
        self.assertEqual(_day_month(date(2026, 7, 14), "italian"), "14 lug")
        self.assertEqual(_display_meal_name("Breakfast", "italian"), "Colazione")
        self.assertEqual(_days_with_data(7, "italian"), "7 giorni con dati")
        self.assertEqual(_excluded_days(1, "italian"), "1 giorno escluso")
        self.assertEqual(_format_health_value("steps", 14200), "14200")
        self.assertEqual(_format_health_value("sleep", 7.2), "7.2 h")


if __name__ == "__main__":
    unittest.main()
