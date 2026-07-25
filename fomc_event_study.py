"""
FOMC Event Study
================

Does implied volatility behave differently around Federal Reserve announcements?

This is an event study, a standard econometric method for measuring how a market
variable responds to scheduled events. The Federal Open Market Committee (FOMC)
sets U.S. monetary policy at eight regularly scheduled meetings per year, with
announcements released at 2:00 PM Eastern Time on the second day. These are
some of the most anticipated events on the financial calendar.

The research question, plainly: does the volatility risk premium found in the
main analysis have a specific, calendar-driven structure? If VIX systematically
rises before FOMC meetings and falls after, then some of that ~4-point premium
is compensation for a scheduled, predictable risk, not diffuse insurance
against unknown shocks. That distinction changes how the premium should be
interpreted.

The empirical approach:

    1. Assemble the FOMC decision-day calendar since 2010.
    2. For each meeting, extract VIX changes in a window from t-3 to t+3.
    3. Test whether the average change on each event day is distinct from zero.
    4. Run an OLS regression with FOMC dummies plus a mean-reversion control,
       using HAC (Newey-West) standard errors.

Data limitation, worth stating up front: the academic literature that finds a
strong FOMC effect uses INTRADAY data, measuring VIX changes in the 30 minutes
around the 2 PM announcement. Free financial data only gives daily closes. A
full trading day contains many other shocks besides the Fed, so any signal here
has to fight through more noise. If a robust FOMC effect shows up at the daily
frequency, that is real. If it does not, that is not evidence the effect does
not exist. It is evidence that daily data is not sharp enough to detect it.

Usage:  python fomc_event_study.py
Output: charts/fomc_window.png, charts/fomc_cumulative.png,
        results/fomc_summary.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

warnings.filterwarnings("ignore")

START = "2010-01-01"
WINDOW = range(-3, 4)              # trading days from t-3 to t+3
CHART_DIR = "charts"
RESULT_DIR = "results"
INK, ACCENT, ACCENT_2 = "#1a1a1a", "#0b5d8a", "#b5451c"
GRID = "#d9d9d9"


# FOMC decision-day calendar since 2010. Second (announcement) day of each
# two-day meeting, sourced from federalreserve.gov historical calendars.
# Includes the emergency inter-meeting decisions of March 3 and March 15, 2020
# during the COVID response, both of which moved markets substantially.
FOMC_DATES_RAW = """
2010-01-27,2010-03-16,2010-04-28,2010-06-23,2010-08-10,2010-09-21,2010-11-03,2010-12-14,
2011-01-26,2011-03-15,2011-04-27,2011-06-22,2011-08-09,2011-09-21,2011-11-02,2011-12-13,
2012-01-25,2012-03-13,2012-04-25,2012-06-20,2012-08-01,2012-09-13,2012-10-24,2012-12-12,
2013-01-30,2013-03-20,2013-05-01,2013-06-19,2013-07-31,2013-09-18,2013-10-30,2013-12-18,
2014-01-29,2014-03-19,2014-04-30,2014-06-18,2014-07-30,2014-09-17,2014-10-29,2014-12-17,
2015-01-28,2015-03-18,2015-04-29,2015-06-17,2015-07-29,2015-09-17,2015-10-28,2015-12-16,
2016-01-27,2016-03-16,2016-04-27,2016-06-15,2016-07-27,2016-09-21,2016-11-02,2016-12-14,
2017-02-01,2017-03-15,2017-05-03,2017-06-14,2017-07-26,2017-09-20,2017-11-01,2017-12-13,
2018-01-31,2018-03-21,2018-05-02,2018-06-13,2018-08-01,2018-09-26,2018-11-08,2018-12-19,
2019-01-30,2019-03-20,2019-05-01,2019-06-19,2019-07-31,2019-09-18,2019-10-30,2019-12-11,
2020-01-29,2020-03-03,2020-03-15,2020-04-29,2020-06-10,2020-07-29,2020-09-16,2020-11-05,2020-12-16,
2021-01-27,2021-03-17,2021-04-28,2021-06-16,2021-07-28,2021-09-22,2021-11-03,2021-12-15,
2022-01-26,2022-03-16,2022-05-04,2022-06-15,2022-07-27,2022-09-21,2022-11-02,2022-12-14,
2023-02-01,2023-03-22,2023-05-03,2023-06-14,2023-07-26,2023-09-20,2023-11-01,2023-12-13,
2024-01-31,2024-03-20,2024-05-01,2024-06-12,2024-07-31,2024-09-18,2024-11-07,2024-12-18,
2025-01-29,2025-03-19,2025-05-07,2025-06-18,2025-07-30,2025-09-17,2025-10-29,2025-12-10,
2026-01-28,2026-03-18,2026-04-29,2026-06-17
"""


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------

def load_vix() -> pd.Series:
    """Daily VIX closing values since 2010."""
    v = yf.download("^VIX", start=START, progress=False, auto_adjust=True)["Close"]
    v.columns = ["VIX"]
    return v.VIX


def fomc_calendar(vix_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Parse the meeting calendar and keep only dates the market was open."""
    parsed = pd.to_datetime([
        s.strip() for s in FOMC_DATES_RAW.replace("\n", ",").split(",") if s.strip()
    ])
    return parsed[parsed.isin(vix_index)]


# ----------------------------------------------------------------------------
# 2. Event window
# ----------------------------------------------------------------------------

def build_event_panel(vix: pd.Series, fomc: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each FOMC meeting, extract daily VIX changes from t-3 to t+3.

    Long form: one row per (meeting, event_day) pair. This is what event study
    regressions and per-day t-tests expect.
    """
    dvix = vix.diff()
    rows = []
    for date in fomc:
        idx = vix.index.get_loc(date)
        for offset in WINDOW:
            pos = idx + offset
            if 0 <= pos < len(vix):
                rows.append({
                    "meeting": date,
                    "event_day": offset,
                    "dvix": dvix.iloc[pos],
                    "vix_lag": vix.iloc[pos - 1] if pos > 0 else np.nan,
                })
    return pd.DataFrame(rows).dropna()


def per_day_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each day in the event window, test whether mean VIX change is nonzero.

    A t-statistic above ~2 (in absolute value) corresponds to p < 0.05 for a
    two-sided test, meaning we would reject "mean is zero" at conventional
    significance. The cumulative_mean column shows the running total, which
    reveals path-dependent effects the point statistics can miss.
    """
    grouped = panel.groupby("event_day").dvix.agg(["mean", "std", "count"])
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["t_stat"] = grouped["mean"] / grouped["se"]
    grouped["cumulative_mean"] = grouped["mean"].cumsum()
    return grouped


# ----------------------------------------------------------------------------
# 3. Full regression with mean-reversion control
# ----------------------------------------------------------------------------

def event_regression(vix: pd.Series, fomc: pd.DatetimeIndex):
    """
    Regress daily VIX changes on FOMC dummy variables plus a control for the
    VIX level going in:

        dVIX_t = a + b1*fomc_t + b2*pre_fomc_t + b3*post_fomc_t + b4*VIX_{t-1} + e

    The lagged-level control captures mean reversion: when VIX is elevated it
    tends to fall the next day regardless of the calendar, so any raw FOMC
    effect would be confounded with that mechanical tendency.
    """
    dvix = vix.diff()
    fomc_flag = pd.Series(vix.index.isin(fomc).astype(int), index=vix.index)

    X = pd.DataFrame({
        "const": 1,
        "fomc_day": fomc_flag,
        "day_before_fomc": fomc_flag.shift(-1).fillna(0).astype(int),
        "day_after_fomc": fomc_flag.shift(1).fillna(0).astype(int),
        "vix_lag": vix.shift(1),
    }).dropna()
    y = dvix.reindex(X.index)

    return sm.OLS(y, X).fit()


# ----------------------------------------------------------------------------
# 4. Charts
# ----------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def plot_window(stats: pd.DataFrame, path: str):
    """Bar chart of mean VIX change per event day, with 95% confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 5))
    days = stats.index.astype(int)
    means = stats["mean"].values
    errs = 1.96 * stats["se"].values

    colors = [ACCENT if m > 0 else ACCENT_2 for m in means]
    ax.bar(days, means, yerr=errs, color=colors, alpha=0.85, width=0.6,
           error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=4))
    ax.axhline(0, color=INK, linewidth=0.9)

    ax.set_xticks(days)
    ax.set_xticklabels([f"t{d:+d}" if d != 0 else "meeting" for d in days])
    ax.set_ylabel("Mean daily change in VIX (points)", fontsize=9.5, color=INK)
    ax.set_title("Volatility Around FOMC Announcements",
                 fontsize=13, color=INK, loc="left", pad=12)
    _style(ax)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cumulative(stats: pd.DataFrame, path: str):
    """Cumulative running total of VIX changes across the window."""
    fig, ax = plt.subplots(figsize=(10, 5))
    days = stats.index.astype(int)
    cum = stats["cumulative_mean"].values

    ax.plot(days, cum, "o-", color=ACCENT, lw=1.6, ms=6)
    ax.axhline(0, color=INK, linewidth=0.9, alpha=0.6)
    ax.axvline(0, color=ACCENT_2, linewidth=1, ls="--", alpha=0.6)
    ax.annotate("meeting", xy=(0, ax.get_ylim()[1]), xytext=(4, -14),
                textcoords="offset points", fontsize=9, color=ACCENT_2)

    ax.set_xticks(days)
    ax.set_xticklabels([f"t{d:+d}" if d != 0 else "0" for d in days])
    ax.set_ylabel("Cumulative VIX change since t-3 (points)",
                  fontsize=9.5, color=INK)
    ax.set_title("Cumulative VIX Drift Around FOMC Meetings",
                 fontsize=13, color=INK, loc="left", pad=12)
    _style(ax)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Run
# ----------------------------------------------------------------------------

def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading VIX and FOMC calendar...")
    vix = load_vix()
    fomc = fomc_calendar(vix.index)
    print(f"  VIX: {len(vix):,} trading days")
    print(f"  FOMC decisions in sample: {len(fomc)}\n")

    print("Building event windows (t-3 to t+3)...")
    panel = build_event_panel(vix, fomc)
    stats = per_day_stats(panel)
    print(stats.round(3))
    print()

    # Highlight the significant days
    sig_days = stats[stats.t_stat.abs() > 1.96]
    if not sig_days.empty:
        print("Days with statistically significant VIX moves (t > 1.96):")
        for day, row in sig_days.iterrows():
            print(f"  t={day:+d}: mean = {row['mean']:+.3f} pts, t = {row['t_stat']:+.2f}")
    else:
        print("No individual day meets the p < 0.05 threshold.")
    print()

    print("Running OLS regression with mean-reversion control...")
    reg = event_regression(vix, fomc)
    print(reg.summary())

    plot_window(stats, f"{CHART_DIR}/fomc_window.png")
    plot_cumulative(stats, f"{CHART_DIR}/fomc_cumulative.png")

    with open(f"{RESULT_DIR}/fomc_summary.txt", "w") as f:
        f.write("PER-DAY EVENT STATISTICS\n")
        f.write("=" * 70 + "\n")
        f.write(stats.round(4).to_string())
        f.write("\n\nOLS REGRESSION: dVIX = a + b*fomc_dummies + c*vix_lag\n")
        f.write("=" * 70 + "\n")
        f.write(str(reg.summary()))

    print(f"\nWrote {CHART_DIR}/ and {RESULT_DIR}/fomc_summary.txt")
    return stats, reg


if __name__ == "__main__":
    main()
