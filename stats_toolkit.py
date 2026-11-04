"""
Statistical Analysis Toolkit

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""


import csv
import math
from pathlib import Path

# Anchor paths on the script's own folder so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "agent_deposits.csv"
OUTPUT_PATH = BASE_DIR / "output" / "stats_report.txt"

# Columns used to identify a row in the outlier list
LABEL_COLUMNS = ("date", "agent_id")


## cleaning data

def _read_column(path: Path, column: str, label_columns=()):
    """Single pass over the CSV.

    Returns (valid_values, labels, rejected):
      valid_values - finite floats from `column`
      labels       - one tuple of `label_columns` values per valid value
      rejected     - (line_number, raw_value, reason) for every unusable value
    Line numbers come from the csv reader, so blank lines don't shift them.
    """
    valid_values, labels, rejected = [], [], []

    try:
        f = open(path, newline="")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}") from None

    with f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in (column, *label_columns) if c not in fields]
        if missing:
            raise ValueError(f"Column(s) {missing} not found in {path}. Available: {fields}")

        for row in reader:
            line_number = reader.line_num
            raw_data = row[column] or ""  # short rows give None

            try:
                number = float(raw_data)
            except ValueError:
                reason = "missing value" if raw_data.strip() == "" else "not a number"
                rejected.append((line_number, raw_data, reason))
                continue

            # float() happily accepts "nan" and "inf", which would break every statistic
            if not math.isfinite(number):
                rejected.append((line_number, raw_data, "not a finite number"))
                continue

            valid_values.append(number)
            labels.append(tuple(row[c] for c in label_columns))

    return valid_values, labels, rejected


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value, reason) tuples."""
    valid_values, _, rejected = _read_column(path, column)
    return valid_values, rejected


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot find mean of empty list!")
    return sum(values) / len(values)


def _median_sorted(s: list[float]) -> float:
    if not s:
        raise ValueError("cannot take the median of an empty list")
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def median(values: list[float]) -> float:
    return _median_sorted(sorted(values))


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""

    # if no value is passed
    if not values:
        raise ValueError("need at least one value to find the mode")

    # count every time a value is found, start count at 0, add 1 for each find
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    highest = max(counts.values())
    # if all values only appear once, no mode, return empty list
    if highest == 1:
        return []

    # return value if it has the highest count
    return [value for value, count in counts.items() if count == highest]


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    n = len(values)
    if n < (2 if sample else 1):
        raise ValueError("Need at least 2 values to calculate variance!")
    data_mean = mean(values)
    # square each difference from the mean (makes it positive), then sum
    squared_diffs = sum((v - data_mean) ** 2 for v in values)
    return squared_diffs / (n - 1 if sample else n)


def std_dev(values: list[float], sample: bool = True) -> float:
    # std dev is the square root of variance, so use power 0.5
    return variance(values, sample) ** 0.5


def _quartiles_sorted(s: list[float]) -> tuple[float, float, float]:
    if len(s) < 2:
        raise ValueError("need at least 2 values to calculate quartiles")
    half = len(s) // 2
    return _median_sorted(s[:half]), _median_sorted(s), _median_sorted(s[len(s) - half:])


def quartiles(values: list[float]) -> tuple[float, float, float]:
    return _quartiles_sorted(sorted(values))


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return Q1, Q3, the IQR, the lower and upper fences, and the values outside them."""
    q1, _, q3 = quartiles(values)
    iqr = q3 - q1
    lower_limit = q1 - k * iqr
    upper_limit = q3 + k * iqr

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "outliers": [v for v in values if v < lower_limit or v > upper_limit],
    }


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values, name="values", rejected=None, labels=None, k=1.5):
        self.name = name
        self.rejected = rejected or []
        self.values = list(values)
        self.labels = list(labels) if labels else [None] * len(self.values)
        self.k = k

        s = sorted(self.values)  # sorted once, used for everything below
        fences = iqr_outliers(s, k)
        lower, upper = fences["lower_limit"], fences["upper_limit"]

        self._outliers = [(v, lab) for v, lab in zip(self.values, self.labels)
                          if v < lower or v > upper]

        # every figure comes from the module-level functions, so there is one definition of each
        self._summary = {
            "count": len(s),
            "mean": mean(s),
            "median": _median_sorted(s),
            "mode": mode(s),
            "variance": variance(s),
            "std_dev": std_dev(s),
            "min": s[0],
            "max": s[-1],
            "q1": fences["q1"],
            "q3": fences["q3"],
            "iqr": fences["iqr"],
            "lower_fence": lower,
            "upper_fence": upper,
            "outliers": [v for v, _ in self._outliers],
        }

    @classmethod
    def from_csv(cls, path, column, label_columns=LABEL_COLUMNS, k=1.5):
        values, labels, rejected = _read_column(path, column, label_columns)
        return cls(values, column, rejected, labels, k)

    def summary(self):
        return dict(self._summary)

    def without_outliers(self):
        s = self._summary
        kept = [(v, lab) for v, lab in zip(self.values, self.labels)
                if s["lower_fence"] <= v <= s["upper_fence"]]
        if len(kept) < 2:
            raise ValueError("fewer than 2 values remain after removing outliers")
        return StatisticsAnalyzer([v for v, _ in kept], self.name, self.rejected,
                                  [lab for _, lab in kept], self.k)

    def report(self):
        def column(s):
            if s is None:
                return ["n/a"] * 9
            return [
                str(s["count"]),
                f"{s['mean']:,.2f}",
                f"{s['median']:,.2f}",
                ", ".join(f"{m:,.2f}" for m in s["mode"]) or "none",
                f"{s['variance']:,.2f}",
                f"{s['std_dev']:,.2f}",
                f"{s['min']:,.2f} / {s['max']:,.2f}",
                f"{s['q1']:,.2f} / {s['q3']:,.2f}",
                f"{s['iqr']:,.2f}",
            ]

        labels = ["count", "mean", "median", "mode", "variance (sample)",
                  "std dev (sample)", "min / max", "Q1 / Q3", "IQR"]

        try:
            trimmed = self.without_outliers().summary()
        except ValueError:
            trimmed = None  # too little data left to summarise

        a, b = column(self._summary), column(trimmed)
        width = max(len(x) for x in a + ["all data"]) + 2

        lines = [
            f"=== Statistics report: {self.name} ===",
            f"Rows read         : {len(self.values) + len(self.rejected)}",
            f"Valid values      : {len(self.values)}",
            f"Rejected values   : {len(self.rejected)}",
        ]
        lines += [f"  row {r}: {raw!r} ({reason})" for r, raw, reason in self.rejected]
        lines += ["", " " * 20 + "all data".ljust(width) + "outliers removed"]
        lines += [l.ljust(20) + x.ljust(width) + y for l, x, y in zip(labels, a, b)]

        s = self._summary
        outliers = ", ".join(
            f"{v:,.2f} ({', '.join(str(x) for x in lab)})" if lab else f"{v:,.2f}"
            for v, lab in self._outliers
        )
        lines += [
            "",
            f"Outlier limits      : lower {s['lower_fence']:,.2f}, upper {s['upper_fence']:,.2f}",
            f"Outliers            : {outliers or 'none'}",
        ]
        return "\n".join(lines)


def compare_with_numpy(analyzer):
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    try:
        import numpy as np
    except ImportError:
        return None

    # v is the data being analysed, s is the summary from our own functions
    v, s = analyzer.values, analyzer.summary()

    # since i am finding quartiles by checking  "median of each half", 
    # the middle value is excluded from both halves when n is odd.
    # TO use NumPy to reproduces this, use the 'weibull' method for odd n and the 'hazen' method for even n. 

    method = "weibull" if len(v) % 2 else "hazen"
    q1, q3 = np.percentile(v, [25, 75], method=method)

    return {
        "mean": np.allclose(s["mean"], np.mean(v)),
        "median": np.allclose(s["median"], np.median(v)),
        # ddof=1 gives the sample (n - 1) variance
        "variance": np.allclose(s["variance"], np.var(v, ddof=1)),
        "std_dev": np.allclose(s["std_dev"], np.std(v, ddof=1)),
        "q1_q3": np.allclose([s["q1"], s["q3"]], [q1, q3]),
    }


def numpy_report(analyzer) -> str:
    """Return the NumPy verification results as text."""
    results = compare_with_numpy(analyzer)
    header = f"=== NumPy verification: {analyzer.name} ==="
    if results is None:
        return f"{header}\nskipped (NumPy not installed)"

    names = {"mean": "mean", "median": "median", "variance": "variance",
             "std_dev": "std dev", "q1_q3": "Q1 / Q3"}
    lines = [header]
    lines += [f"{names[key]:<10}{'MATCH' if ok else 'MISMATCH'}"
              for key, ok in results.items()]
    return "\n".join(lines)


def main() -> None:
    deposits = StatisticsAnalyzer.from_csv(DATA_PATH, "daily_deposits_kes")
    text = "\n\n".join([deposits.report(), numpy_report(deposits)]) + "\n"

    print(text)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # create output/ if it is missing
    OUTPUT_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()