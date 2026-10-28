# %%
"""
Statistical Analysis Toolkit

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"

OUTPUT_PATH = Path("output") / "stats_report.txt"

# %% [markdown]
# ---

# %% [markdown]
# ## cleaning data

# %%
def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    valid_values, rejected = [], []

    # importing data using dict reader
    with open(path, newline="") as f:
        for index, row in enumerate(csv.DictReader(f), start=2):
            raw_data = row[column]
            try:
                # attmept to convert col to float
                valid_values.append(float(raw_data))
            except ValueError:
                # if float conversion fails, find reason 
                reason = "missing value" if raw_data.strip() == "" else "not a number"
                rejected.append((index, raw_data, reason)) 

    return valid_values, rejected


# viewing unusable data in daily deposit and transaction cols
valid_deposits, rejected_deposits = load_column(DATA_PATH, "daily_deposits_kes")

print(rejected_deposits)

valid_txns, rejected_txns = load_column(DATA_PATH, "transactions")
print(rejected_txns)

# print(valid_deposits)
# 6 columns unuseable

# %% [markdown]
# ---

# %%
def mean(values: list[float]) -> float:
    return sum(values) / len(values)


# finding mean of valid deposits
deposits_mean = mean(valid_deposits)
print("Mean of valid deposits")
print(f"{deposits_mean:,.2f}")
print()

# finding mean of valid transactions
print("Mean of vlaid txns")
txns_mean = mean(valid_txns)
print(f"{txns_mean:,.2f}")

# %%
def _median_sorted(s: list[float]) -> float:
    if not s:
        raise ValueError("cannot take the median of an empty list")
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

def median(values: list[float]) -> float:
    return _median_sorted(sorted(values))

# finding median of valid deposits
deposits_median = median(valid_deposits)
print("Deposits median")
print(deposits_median)
print()

# finding meadian of valid txns
print("Txns median")
txns_median = median(valid_txns)
print(txns_median)


# %%
def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    # initialize empty set counts to hold unique values
    counts = {}

    # count every time a value is found, start count at 0, add 1 for each find
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    # if no value is found, no mode, return empty list
    if not counts:
        return []

    # find highest count
    highest = max(counts.values())
    # if all values only appear once, no mode, return empty list
    if highest == 1:
        return []

    # return value if it has the highest count
    return [value for value, count in counts.items() if count == highest]


# finding mode
# finding txns mode
txns_mode = mode(valid_txns)
print("Txns Mode")
print(txns_mode)



# %%
def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    # finding count of values passed
    n = len(values)
    # if < 2 values
    if n < (2 if sample else 1):
        raise ValueError("Need at least 2 values to calculate vaariance!")
    data_mean = mean(values)
    # find diff of value - mean
    # square diff, to make it posive
    # sum all +ve squared values
    squared_diffs = sum((v- data_mean) ** 2 for v in values)
    # divide sum by count - 1
    return squared_diffs / (n - 1 if sample else n)


# test variance
deposits_var = variance(valid_deposits)
print("Deposits Variance")
print(deposits_var)
print()

txns_var = variance(valid_txns)
print("Txns variance")
print(txns_var)

# %%
def std_dev(values: list[float], sample: bool = True) -> float:
    # std dev id square root of variance so use power 0.5 to find square root
    return variance(values, sample) ** 0.5


# test std dev
deposits_std_dev = std_dev(valid_deposits)
print("Deposits std dev")
print(deposits_std_dev)
print()

txns_std_dev = std_dev(valid_txns)
print("Txns std dev")
print(txns_std_dev)

# %%
def _quartiles_sorted(s: list[float]) -> tuple[float, float, float]:
    if len(s) < 2:
        raise ValueError("need at least 2 values to calculate quartiles")
    half = len(s) // 2
    return _median_sorted(s[:half]), _median_sorted(s), _median_sorted(s[len(s) - half:])

def quartiles(values: list[float]) -> tuple[float, float, float]:
    return _quartiles_sorted(sorted(values))


# testing quartiles
deposits_quats = quartiles(valid_deposits)
print("deposits quartiles")
print(deposits_quats)
print()

txns_quats = quartiles(valid_txns)
print("txns quartiles")
print(txns_quats)
print()

# %%
def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    q1, _, q3 = quartiles(values)
    iqr = q3 - q1
    lower_limit = q1 - k * iqr
    upper_limit = q3 + k * iqr

    return {
        "iqr" : iqr,
        "lower_limit" : lower_limit,
        "upper_limit": upper_limit,
        # return value from values
        "outliers" : [v for v in values 
                      # if value < or > than limits
                      if v < lower_limit 
                      or v > upper_limit],
    }


# testing IQR
deposits_IQR = iqr_outliers(valid_deposits)
print("deposits outliers")
print(deposits_IQR)
print()

txns_IQR = iqr_outliers(valid_txns)
print("txns quartiles")
print(txns_IQR)
print()

# %% [markdown]
# ---

# %%
class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values, name="values", rejected=None, labels=None):
        self.name = name
        self.rejected = rejected or []
        self.values = list(values)
        self.labels = labels or [None] * len(values)
        s = sorted(self.values)  # sorted once, used for everything below
        n = len(s)
        q1, median, q3 = _quartiles_sorted(s)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        var = variance(s)

        self._outliers = [(v, lab) for v, lab in zip(self.values, self.labels)
                          if v < lower or v > upper]
        
        self._summary = {
            "count": n, "mean": sum(s) / n, "median": median, "mode": mode(s),
            "variance": var, "std_dev": var ** 0.5, "min": s[0], "max": s[-1],
            "q1": q1, "q3": q3, "iqr": iqr, "lower_fence": lower, "upper_fence": upper,
            "outliers": [v for v, _ in self._outliers],
        }

    @classmethod
    def from_csv(cls, path, column):
        values, rejected = load_column(path, column)
        bad_rows = {row_number for row_number, _, _ in rejected}
        with open(path, newline="") as f:
            labels = [(row["date"], row["agent_id"])
                      for i, row in enumerate(csv.DictReader(f), start=2) if i not in bad_rows]
        return cls(values, column, rejected, labels)

    def summary(self):
        return dict(self._summary)

    def without_outliers(self):
        s = self._summary
        kept = [(v, lab) for v, lab in zip(self.values, self.labels)
                if s["lower_fence"] <= v <= s["upper_fence"]]
        return StatisticsAnalyzer([v for v, _ in kept], self.name, self.rejected,
                                  [lab for _, lab in kept])

    def report(self):
        def column(s):
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
        a, b = column(self._summary), column(self.without_outliers()._summary)
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
            f"{v:,.2f} ({lab[0]}, {lab[1]})" if lab else f"{v:,.2f}"
            for v, lab in self._outliers
        )
        lines += [
            "",
            f"Outlier fences      : lower {s['lower_fence']:,.2f}, upper {s['upper_fence']:,.2f}",
            f"Outliers            : {outliers or 'none'}",
        ]
        return "\n".join(lines)

    

# %%
transactions = StatisticsAnalyzer.from_csv(DATA_PATH, "transactions")
print(transactions.report())
print()


# %%
deposits = StatisticsAnalyzer.from_csv(DATA_PATH, "daily_deposits_kes")
print(deposits.report())


# %% [markdown]
# ---

# %%
def compare_with_numpy(analyzer):
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np

    # v is set data to be analysed using numpy
    # s is set data from summary func
    v, s = analyzer.values, analyzer.summary()

    # if dataset length even use weibull to find percentiles else use hazen method
    method = "weibull" if len(v) % 2 else "hazen"
    # finding percentiles
    q1, q3 = np.percentile(v, [25, 75], method=method)

    #using allclose to compare values in the s and v data sets
    results = {
        "mean": np.allclose(s["mean"], np.mean(v)),
        "median": np.allclose(s["median"], np.median(v)),
        # using ddof = 1 to use n-1 to calculate variance
        "variance": np.allclose(s["variance"], np.var(v, ddof=1)),
        "std_dev": np.allclose(s["std_dev"], np.std(v, ddof=1)),
        "q1_q3": np.allclose([s["q1"], s["q3"]], [q1, q3]),
    }

    def word(key):
        return "MATCH" if results[key] else "MISMATCH"

    print("=== NumPy verification ===")
    print(f"mean      {word('mean')}")
    print(f"median    {word('median')}")
    print(f"variance  {word('variance')} ")
    print(f"std dev   {word('std_dev')} ")
    print(f"Q1 / Q3   {word('q1_q3')} ")
    return results

# %%
# test compare with numpy
analyzer = compare_with_numpy(deposits)

# %%
def main() -> None:
    columns = ["daily_deposits_kes", "transactions"]
    reports = [StatisticsAnalyzer.from_csv(DATA_PATH, c).report() for c in columns]
    text = "\n\n".join(reports) + "\n"

    print(text)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)  # create output/ if it is missing
    OUTPUT_PATH.write_text(text, encoding="utf-8")


# %%
if __name__ == "__main__":
    main()



