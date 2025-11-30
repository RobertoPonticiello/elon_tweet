from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from statsmodels.discrete.discrete_model import NegativeBinomial
import statsmodels.api as sm
from pandas.api import types as ptypes

sns.set_theme(style="whitegrid")

WINDOW_START = pd.Timestamp("2025-11-25", tz="UTC")
WINDOW_END = pd.Timestamp("2025-12-02", tz="UTC")
TIME_REMAINING = pd.Timedelta(days=4, hours=2)
KNOWN_TOTAL = 135  # posts already counted inside the market window

window_end_exclusive = WINDOW_END + pd.Timedelta(days=1)
CURRENT_TIMESTAMP = window_end_exclusive - TIME_REMAINING
if CURRENT_TIMESTAMP.tzinfo is None:
    CURRENT_TIMESTAMP = CURRENT_TIMESTAMP.tz_localize("UTC")

print(
    f"Using implied current timestamp {CURRENT_TIMESTAMP:%Y-%m-%d %H:%M %Z}"
    f" based on {TIME_REMAINING} remaining until end-of-day {WINDOW_END.date()}."
)

MANUAL_PARSE_NAME = "elonmusk (2).csv"
SOURCE_TZ = "America/New_York"
DATA_CANDIDATES = [Path(MANUAL_PARSE_NAME), Path("elon_posts.csv"), Path("all_musk_posts.csv")]
for candidate in DATA_CANDIDATES:
    if candidate.exists():
        DATA_PATH = candidate
        break
else:
    raise FileNotFoundError(
        "Could not find elonmusk (2).csv, elon_posts.csv, or all_musk_posts.csv in the workspace."
    )

print(f"Loading data from {DATA_PATH} ...")
def _parse_three_field_csv(path: Path) -> pd.DataFrame:
    text = path.read_text()
    newline_idx = text.find("\n")
    if newline_idx == -1:
        raise ValueError("CSV appears to have no newline/header row")

    records: List[Tuple[str, str, str]] = []
    i = newline_idx + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in "\r\n":
            i += 1
        if i >= n:
            break

        start_i = i
        while i < n and text[i] != ",":
            i += 1
        if i >= n:
            break
        post_id = text[start_i:i]
        i += 1

        if i < n and text[i] == '"':
            i += 1
        body_chars: List[str] = []
        while i < n:
            if text[i] == '"' and i + 2 < n and text[i + 1] == ',' and text[i + 2] == '"':
                i += 3
                break
            body_chars.append(text[i])
            i += 1
        body = ''.join(body_chars)

        created_chars: List[str] = []
        while i < n and text[i] != '"':
            created_chars.append(text[i])
            i += 1
        created_raw = ''.join(created_chars)
        if i < n and text[i] == '"':
            i += 1

        while i < n and text[i] in "\r\n":
            i += 1

        records.append((post_id, body, created_raw))

    return pd.DataFrame(records, columns=["id", "text", "created_at"])

if DATA_PATH.name == MANUAL_PARSE_NAME:
    df = _parse_three_field_csv(DATA_PATH)
    print(f"Parsed {len(df):,} rows from manual three-field CSV")
    dt_series = pd.to_datetime(
        df["created_at"].str.strip() + " 2025",
        errors="coerce",
    )
    if dt_series.isna().any():
        missing = int(dt_series.isna().sum())
        raise ValueError(f"Failed to parse {missing} timestamps from {DATA_PATH.name}")
    dt_series = dt_series.dt.tz_localize(
        SOURCE_TZ,
        ambiguous="infer",
        nonexistent="shift_forward",
    ).dt.tz_convert("UTC")
    df["createdAt"] = dt_series
    print("\nAvailable columns:\n", df.columns.tolist())
else:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows")
    print("\nAvailable columns:\n", df.columns.tolist())

# --- Helper utilities -----------------------------------------------------
def str_to_bool(series: pd.Series) -> pd.Series:
    """Convert string/object True/False/NaN flags to booleans."""
    lowered = series.astype(str).str.lower()
    return lowered == "true"

# --- Task 1: Market rule filters -----------------------------------------
# Rule mapping notes:
# 1. Main feed posts -> rows where isReply flag is False. Dataset exposes this in `isReply`.
# 2. Quote posts -> `isQuote == True`.
# 3. Reposts/retweets -> `isRetweet == True`.
# 4. Replies should be excluded (isReply == True) unless we can detect "main feed replies".
#    The export does not provide a dedicated flag for that special case, so this rule is
#    approximated by removing all rows with isReply == True but keeping quotes/retweets even
#    if their reply flag is mislabeled.
# 5. Deleted posts: anything present in the export is assumed to have lived long enough, so
#    no additional filter is applied.
# 6. Community reposts: there is no explicit indicator in the export, so they cannot be
#    filtered separately and remain included by necessity.

bool_cols = {"isReply", "isRetweet", "isQuote"}
if bool_cols.issubset(df.columns):
    df["is_reply"] = str_to_bool(df["isReply"])
    df["is_retweet"] = str_to_bool(df["isRetweet"])
    df["is_quote"] = str_to_bool(df["isQuote"])

    main_feed_mask = ~df["is_reply"]
    quote_mask = df["is_quote"]
    retweet_mask = df["is_retweet"]

    valid_mask = main_feed_mask | quote_mask | retweet_mask
    filtered = df.loc[valid_mask].copy()
    print(f"\nFiltered down to {len(filtered):,} rows (from {len(df):,}).")
    print(filtered[["is_reply", "is_quote", "is_retweet"]].head())
else:
    print("\nDataset lacks reply/quote/retweet flags; using all rows without additional filtering.")
    filtered = df.copy()

# --- Task 2+: Daily counts + modeling sensitivity -----------------------
if ptypes.is_datetime64tz_dtype(filtered["createdAt"]):
    filtered["createdAt"] = filtered["createdAt"].dt.tz_convert("UTC")
else:
    filtered["createdAt"] = pd.to_datetime(filtered["createdAt"], utc=True, errors="coerce")
filtered = filtered.dropna(subset=["createdAt"]).sort_values("createdAt")
global_latest_timestamp = filtered["createdAt"].max()
print(f"Latest captured post timestamp: {global_latest_timestamp:%Y-%m-%d %H:%M %Z}")

WINDOW_ORDER = [
    "Last 1 month",
    "Last 3 months",
    "Last 6 months",
    "From Jan 1 2025",
]
WINDOW_SPEC_CANDIDATES = [
    ("Last 1 month", global_latest_timestamp - pd.DateOffset(months=1)),
    ("Last 3 months", global_latest_timestamp - pd.DateOffset(months=3)),
    ("Last 6 months", global_latest_timestamp - pd.DateOffset(months=6)),
    ("From Jan 1 2025", pd.Timestamp("2025-01-01", tz="UTC")),
]
data_floor = filtered["createdAt"].min()
WINDOW_SPECS = []
for label, start in WINDOW_SPEC_CANDIDATES:
    adjusted_start = max(start, data_floor)
    WINDOW_SPECS.append((label, adjusted_start))
range_bins = [(start, start + 19) for start in range(100, 420, 20)]
n_sims = 1_000_000

current_day = CURRENT_TIMESTAMP.normalize()
elapsed_seconds = max(0.0, (CURRENT_TIMESTAMP - current_day).total_seconds())
elapsed_seconds = min(elapsed_seconds, 24 * 3600)
remaining_fraction_today = max(0.0, 1.0 - elapsed_seconds / (24 * 3600))

def run_window(label: str, window_start: pd.Timestamp, seed_offset: int) -> dict:
    window_df = filtered[filtered["createdAt"] >= window_start].copy()
    if window_df.empty:
        raise ValueError(f"No data available inside requested window '{label}'.")
    print(
        f"\n=== Training window: {label} (from {window_start.date()} to {global_latest_timestamp.date()}) ==="
    )

    daily_counts = (
        window_df.set_index("createdAt")["id"]
        .resample("D")
        .count()
        .rename("count")
        .to_frame()
        .reset_index()
    )
    print("Daily count sample:")
    print(daily_counts.head())

    plot_slug = (
        label.lower()
        .replace(" ", "_")
        .replace("/", "-")
        .replace(".", "")
    )
    plot_path = Path(f"daily_counts_{plot_slug}.png")
    plt.figure(figsize=(14, 5))
    plt.plot(daily_counts["createdAt"], daily_counts["count"], linewidth=1.0)
    plt.title(f"Posts per day - {label}")
    plt.xlabel("Date")
    plt.ylabel("Posts per day")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Daily counts plot saved to {plot_path.resolve()}")

    latest_daily = daily_counts["createdAt"].max()
    daily_counts["trend"] = (
        daily_counts["createdAt"] - latest_daily
    ).dt.days
    daily_counts["dow"] = daily_counts["createdAt"].dt.dayofweek

    dow_dummies = pd.get_dummies(daily_counts["dow"], prefix="dow", drop_first=True)
    X = pd.concat([daily_counts[["trend"]], dow_dummies], axis=1)
    X = sm.add_constant(X).astype(float)
    y = daily_counts["count"].astype(float)

    nb_model = NegativeBinomial(y, X)
    nb_results = nb_model.fit(disp=False)
    print("Negative Binomial regression summary:")
    print(nb_results.summary())

    baseline_mean = float(np.exp(nb_results.params["const"]))
    alpha = float(nb_results.params["alpha"])
    print(f"Baseline mean daily posts (trend=0, Monday): {baseline_mean:.2f}")
    print(
        "Dispersion alpha: "
        f"{alpha:.3f} (variance = mu + alpha * mu^2; alpha>0 => over-dispersion vs Poisson)"
    )

    forecast_dates = pd.date_range(WINDOW_START, WINDOW_END, freq="D", tz="UTC")
    future_df = pd.DataFrame({"createdAt": forecast_dates})
    future_df["trend"] = (
        future_df["createdAt"] - latest_daily
    ).dt.days.clip(upper=0)
    future_df["dow"] = future_df["createdAt"].dt.dayofweek

    future_dummies = pd.get_dummies(future_df["dow"], prefix="dow", drop_first=True)
    future_X = pd.concat([future_df[["trend"]], future_dummies], axis=1)
    future_X = sm.add_constant(future_X).astype(float)
    future_X = future_X.reindex(columns=X.columns, fill_value=0.0)

    mu_forecast = nb_results.predict(future_X)
    forecast_table = pd.DataFrame(
        {
            "date": forecast_dates,
            "expected_posts": mu_forecast,
        }
    )
    print(f"Forecasted mean posts per day ({WINDOW_START.date()} - {WINDOW_END.date()}):")
    print(forecast_table)

    forecast_table["fraction_remaining"] = 1.0
    forecast_table.loc[forecast_table["date"] < current_day, "fraction_remaining"] = 0.0
    if current_day in set(forecast_table["date"]):
        forecast_table.loc[
            forecast_table["date"] == current_day, "fraction_remaining"
        ] = remaining_fraction_today

    remaining_rows = forecast_table[forecast_table["fraction_remaining"] > 0].copy()
    remaining_rows["adjusted_mu"] = (
        remaining_rows["expected_posts"] * remaining_rows["fraction_remaining"]
    )
    print(
        f"Remaining horizon (conditional on observed {KNOWN_TOTAL} posts through {CURRENT_TIMESTAMP:%Y-%m-%d %H:%M %Z}):"
    )
    print(remaining_rows[["date", "fraction_remaining", "adjusted_mu"]])

    size_param = 1.0 / alpha
    rng = np.random.default_rng(42 + seed_offset)

    def draw_nb(mean: float) -> np.ndarray:
        if mean <= 0:
            return np.zeros(n_sims)
        p = size_param / (size_param + mean)
        return rng.negative_binomial(size_param, p, size=n_sims)

    simulated_counts = (
        np.column_stack([draw_nb(mu) for mu in remaining_rows["adjusted_mu"].to_numpy()])
        if not remaining_rows.empty
        else np.zeros((n_sims, 0))
    )
    scenario_totals = KNOWN_TOTAL + simulated_counts.sum(axis=1)

    prob_rows = []
    for low, high in range_bins:
        prob = np.mean((scenario_totals >= low) & (scenario_totals <= high))
        prob_rows.append({"range": f"{low}-{high}", "probability": prob})

    prob_table = pd.DataFrame(prob_rows)
    print(
        f"Probability of total {WINDOW_START.date()}-{WINDOW_END.date()} posts in each range "
        f"(100k sims, conditional on {KNOWN_TOTAL} already counted):"
    )
    print(prob_table)

    mean_total = scenario_totals.mean()
    median_total = np.median(scenario_totals)
    interval = np.percentile(scenario_totals, [5, 95])
    print(
        f"Simulated total posts mean {mean_total:.1f}, median {median_total:.1f}, "
        f"90% interval [{interval[0]:.1f}, {interval[1]:.1f}] given {KNOWN_TOTAL} already counted."
    )

    return {
        "label": label,
        "cutoff": window_start.date(),
        "nb_results": nb_results,
        "prob_table": prob_table.assign(window_label=label),
        "mean_total": mean_total,
        "median_total": median_total,
        "interval": interval,
    }

scenario_results = [
    run_window(label, start, idx)
    for idx, (label, start) in enumerate(WINDOW_SPECS, start=1)
]

prob_summary = pd.concat([res["prob_table"] for res in scenario_results], ignore_index=True)
prob_pivot = prob_summary.pivot(index="range", columns="window_label", values="probability")
prob_pivot = prob_pivot.sort_index()
ordered_columns = [label for label in WINDOW_ORDER if label in prob_pivot.columns]
prob_pivot = prob_pivot[ordered_columns]

print("\n=== Probability comparison by training window ===")
print(prob_pivot)

summary_rows = []
for res in scenario_results:
    summary_rows.append(
        {
            "window": res["label"],
            "mean_total": res["mean_total"],
            "median_total": res["median_total"],
            "p5": res["interval"][0],
            "p95": res["interval"][1],
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_order = [label for label in WINDOW_ORDER if label in summary_df["window"].values]
summary_df = summary_df.set_index("window").loc[summary_order].reset_index()
print("\n=== Aggregate distribution stats by training window ===")
print(summary_df)

print(f"\nUser-reported running total offset applied: {KNOWN_TOTAL}.")
