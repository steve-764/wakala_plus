"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import tempfile
import unittest
from pathlib import Path

from analysis import (StatisticsAnalyzer, iqr_outliers, load_column, median, mode,
                           quartiles, std_dev, variance)

# Classic textbook set: mean 5, squared differences sum to 32.
SPREAD_DATA = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        summary = StatisticsAnalyzer([2.0, 4.0, 6.0, 8.0]).summary()
        self.assertAlmostEqual(summary["mean"], 5.0)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)        # odd: middle value
        self.assertEqual(median([4.0, 1.0, 3.0, 2.0]), 2.5)   # even: mean of 2 and 3

    def test_mode_with_ties(self):
        self.assertEqual(mode([1.0, 2.0, 2.0, 3.0]), [2.0])              # single mode
        self.assertEqual(sorted(mode([1.0, 1.0, 2.0, 2.0, 3.0])), [1.0, 2.0])  # tie
        self.assertEqual(mode([1.0, 2.0, 3.0]), [])                      # no repeats

    def test_empty_input_raises(self):
        for func in (median, variance, quartiles, StatisticsAnalyzer):
            with self.subTest(func=func.__name__):
                with self.assertRaises(ValueError):
                    func([])
        self.assertEqual(mode([]), [])  # mode of nothing is simply "no mode"


class TestSpread(unittest.TestCase):
    def test_sample_vs_population_variance(self):
        self.assertAlmostEqual(variance(SPREAD_DATA, sample=False), 32 / 8)  # 4.0
        self.assertAlmostEqual(variance(SPREAD_DATA), 32 / 7)                # n - 1
        self.assertAlmostEqual(std_dev(SPREAD_DATA, sample=False), 2.0)
        self.assertAlmostEqual(std_dev(SPREAD_DATA), (32 / 7) ** 0.5)

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([5.0], sample=False), 0.0)
        with self.assertRaises(ValueError):  # sample formula would divide by n - 1 = 0
            std_dev([5.0])


class TestOutliers(unittest.TestCase):
    def test_quartiles_even_and_odd_lengths(self):
        self.assertEqual(quartiles([1, 2, 3, 4, 5]), (1.5, 3, 4.5))      # middle left out
        self.assertEqual(quartiles([1, 2, 3, 4, 5, 6]), (2, 3.5, 5))     # halves of 3

    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(result["iqr"], 4.0)                 # Q3 6.5 - Q1 2.5
        self.assertEqual(result["lower_limit"], -3.5)        # 2.5 - 1.5 * 4
        self.assertEqual(result["upper_limit"], 12.5)        # 6.5 + 1.5 * 4
        self.assertEqual(result["outliers"], [])

    def test_detects_obvious_outlier(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        self.assertEqual(iqr_outliers(data)["outliers"], [100.0])
        cleaned = StatisticsAnalyzer(data).without_outliers().summary()
        self.assertEqual(cleaned["count"], 7)
        self.assertEqual(cleaned["max"], 7.0)


class TestLoading(unittest.TestCase):
    CSV = (
        "date,agent_id,town,daily_deposits_kes,transactions\n"
        "2026-01-01,AG-1,X,100.5,10\n"
        "2026-01-02,AG-1,X,,11\n"
        "2026-01-03,AG-1,X,abc,12\n"
        "2026-01-04,AG-1,X,200,\n"
    )

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "tiny.csv"
        self.path.write_text(self.CSV)

    def test_invalid_and_missing_values_are_rejected(self):
        valid, rejected = load_column(self.path, "daily_deposits_kes")
        self.assertEqual(valid, [100.5, 200.0])
        self.assertEqual(rejected, [(3, "", "missing value"), (4, "abc", "not a number")])

        valid, rejected = load_column(self.path, "transactions")
        self.assertEqual(valid, [10.0, 11.0, 12.0])
        self.assertEqual(rejected, [(5, "", "missing value")])

    def test_report_lists_rejected_rows(self):
        analyzer = StatisticsAnalyzer.from_csv(self.path, "transactions")
        report = analyzer.report()
        self.assertIn("Rows read         : 4", report)
        self.assertIn("Rejected values   : 1", report)
        self.assertIn("row 5: '' (missing value)", report)


if __name__ == "__main__":
    unittest.main()