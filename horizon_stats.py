"""Summarize historical Elon Musk posting horizons for quick reference tables.

This helper ingests the manual CSV export, reconstructs timezone-aware
timestamps, computes descriptive stats for several trailing windows, and writes
the results to ``horizon_summary.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

RAW_PATH = Path("elonmusk (2).csv")
TZ = "America/New_York"
ANALYSIS_REF = pd.Timestamp("2025-11-26 00:00", tz=TZ)


@dataclass
class WindowSpec:
    label: str
    start: pd.Timestamp


def _parse_three_field_csv(path: Path) -> pd.DataFrame:
    """Read custom three-column CSV files that lack standard quoting."""
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
        i += 1  # skip comma

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


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize naive timestamp strings into a timezone-aware column."""
    df = df.copy()
    # created_at strings look like "Jan 1, 12:18:52 AM EST" (no year), so append 2025.
    dt_series = pd.to_datetime(
        df["created_at"].str.strip() + " 2025",
        errors="coerce",
    )
    if dt_series.isna().any():
        missing = dt_series.isna().sum()
        raise ValueError(f"Failed to parse {missing} timestamps from CSV")
    if dt_series.dt.tz is None:
        # Handle DST fallbacks (ambiguous) and spring forwards (nonexistent).
        dt_series = dt_series.dt.tz_localize(
            TZ,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    else:
        dt_series = dt_series.dt.tz_convert(TZ)
    df["created_at_dt"] = dt_series
    return df


def window_stats(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Calculate posting stats for a closed-open window [start, end)."""
    start = pd.Timestamp(start).tz_convert(TZ).floor("D")
    end = pd.Timestamp(end).tz_convert(TZ).ceil("D")
    mask = (df["created_at_dt"] >= start) & (df["created_at_dt"] < end)
    subset = df.loc[mask]
    calendar_index = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="D", tz=TZ)

    if subset.empty:
        calendar_days = len(calendar_index)
        return {
            "posts": 0,
            "calendar_days": calendar_days,
            "active_days": 0,
            "avg_posts_per_day": 0.0,
            "median_posts_per_day": 0.0,
            "avg_posts_per_active_day": 0.0,
            "max_posts_per_day": 0,
            "max_day": None,
        }

    daily_counts = (
        subset.set_index("created_at_dt")["id"]
        .resample("D")
        .count()
        .reindex(calendar_index, fill_value=0)
    )
    calendar_days = len(calendar_index)
    active_days = int((daily_counts > 0).sum())
    avg_per_day = daily_counts.mean()
    median_per_day = daily_counts.median()
    avg_per_active = daily_counts[daily_counts > 0].mean()
    max_posts = int(daily_counts.max())
    max_day_ts = daily_counts.idxmax()
    max_day = max_day_ts.date().isoformat() if isinstance(max_day_ts, pd.Timestamp) else None

    return {
        "posts": int(daily_counts.sum()),
        "calendar_days": calendar_days,
        "active_days": active_days,
        "avg_posts_per_day": avg_per_day,
        "median_posts_per_day": median_per_day,
        "avg_posts_per_active_day": avg_per_active,
        "max_posts_per_day": max_posts,
        "max_day": max_day,
    }


def main() -> None:
    raw_df = _parse_three_field_csv(RAW_PATH)
    df = _coerce_datetimes(raw_df)
    horizon_specs = [
        WindowSpec("From Jan 1 2025", pd.Timestamp("2025-01-01", tz=TZ)),
        WindowSpec("Last 6 months", ANALYSIS_REF - pd.DateOffset(months=6)),
        WindowSpec("Last 3 months", ANALYSIS_REF - pd.DateOffset(months=3)),
        WindowSpec("Last 1 month", ANALYSIS_REF - pd.DateOffset(months=1)),
    ]
    rows = []
    for spec in horizon_specs:
        start = max(spec.start, df["created_at_dt"].min())
        stats = window_stats(df, start, ANALYSIS_REF)
        rows.append({"window": spec.label, **stats})

    summary = pd.DataFrame(rows)
    summary.to_csv("horizon_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))


if __name__ == "__main__":
    main()
