import math
import tempfile
from pathlib import Path

from stats_toolkit import (
    OUTPUT_PATH, StatisticsAnalyzer, iqr_outliers, load_column,
    mean, median, mode, quartiles, std_dev, variance,
)

DATA = [10, 11, 12, 13, 14, 15, 100]  # 100 is an obvious outlier


def test_mean():
    assert mean([1, 2, 3, 4, 5]) == 3


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5


def test_mode():
    assert mode([1, 2, 2, 3]) == [2]
    assert mode([1, 2, 3]) == []


def test_variance():
    assert variance([1, 2, 3, 4, 5]) == 2.5
    assert variance([1, 2, 3, 4, 5], sample=False) == 2


def test_std_dev():
    assert math.isclose(std_dev([1, 2, 3, 4, 5]), 2.5 ** 0.5)


def test_quartiles():
    assert quartiles([1, 2, 3, 4, 5]) == (1.5, 3, 4.5)


def test_iqr_outliers():
    assert iqr_outliers(DATA)["outliers"] == [100]


def test_load_column():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sample.csv"
        path.write_text("a,b\n1,x\n,y\nnan,z\n3,w\n")
        valid, rejected = load_column(path, "a")
    assert valid == [1.0, 3.0]
    assert rejected == [(3, "", "missing value"), (4, "nan", "not a finite number")]


def test_analyzer():
    analyzer = StatisticsAnalyzer(DATA)
    assert analyzer.summary()["outliers"] == [100]
    assert analyzer.without_outliers().summary()["count"] == 6


def run_all():
    """Run every test_ function, return (tests_ran, tests_failed)."""
    tests = [f for name, f in globals().items() if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as error:
            failed += 1
            print(f"FAILED: {test.__name__} ({error!r})")  # detail stays in the terminal
    return len(tests), failed


if __name__ == "__main__":
    ran, failed = run_all()
    marker = "=== Test report ==="
    summary = f"{marker}\nTests ran: {ran}, failed: {failed}"
    print(summary)

    # keep the main report, replace any test report left by an earlier run
    old = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
    base = old.split(marker)[0].rstrip("\n")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text((base + "\n\n" if base else "") + summary + "\n", encoding="utf-8")