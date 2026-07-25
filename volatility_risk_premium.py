"""
Volatility Risk Premium and Forecast Comparison
================================================

The implied volatility work in implied_vol.py answers "what does the market
expect right now." This module asks a harder question: is that expectation
any good? Two tests:

  1. THE PREMIUM     - Does implied volatility (IV) consistently sit above or
                        below what volatility turns out to be? A persistent
                        gap is called the volatility risk premium (VRP).

  2. THE FORECAST     - Is IV actually a better forecast of future volatility
                        than the simplest alternative: just using how volatile
                        the stock has been recently? This is the real test of
                        whether the options market adds information.

Data constraint, stated plainly: free option chain data only gives today's
snapshot, not what implied volatility looked like a year ago. So this uses the
VIX index as the long-history stand-in. VIX is built the same way this
project's implied_vol() function works - inverting option prices to back out
volatility - just aggregated across the S&P 500 rather than one strike, and
published daily back to the 1990s.

Usage:  python volatility_risk_premium.py
Output: charts/vrp_timeseries.png, charts/vrp_calibration.png,
        charts/vrp_distribution.png, results/vrp_summary.txt
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
from scipy import stats

warnings.filterwarnings("ignore")

START = "2010-01-01"
HORIZON = 21          # trading days ~ one month, matching a 30-calendar-day VIX contract

CHART_DIR = "charts"
RESULT_DIR = "results"
INK, ACCENT, ACCENT_2 = "#1a1a1a", "#0b5d8a", "#b5451c"
GRID = "#d9d9d9"


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------

def load_data():
    """
    Match daily SPY closing prices with the VIX index over the same span.

    VIX is quoted as an annualized percentage already, so it sits on the same
    scale as the realized volatility computed below without conversion.
    """
    spy = yf.download("SPY", start=START, progress=False, auto_adjust=True)["Close"]
    vix = yf.download("^VIX", start=START, progress=False, auto_adjust=True)["Close"]
    spy.columns, vix.columns = ["SPY"], ["VIX"]
    return spy.join(vix, how="inner").dropna()


def build_panel(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    For every trading day, line up three numbers:

      iv           - VIX on that day: the market's forecast for the next ~30 days
      fwd_rv       - what volatility actually was, realized over the NEXT `horizon`
                     trading days. This is only knowable in hindsight.
      trailing_rv  - what volatility was over the PAST `horizon` days. This is
                      the naive forecast: "assume tomorrow looks like yesterday."

    fwd_rv and trailing_rv are both annualized standard deviations of daily log
    returns, expressed in percent, so they sit on the same scale as VIX.
    """
    log_ret = np.log(df.SPY / df.SPY.shift(1))
    ann = np.sqrt(252) * 100

    fwd_rv = log_ret.shift(-horizon).rolling(horizon).std() * ann
    fwd_rv = fwd_rv.shift(-(horizon - 1))     # window covers t+1 .. t+horizon
    trailing_rv = log_ret.rolling(horizon).std() * ann

    panel = pd.DataFrame({"iv": df.VIX, "fwd_rv": fwd_rv, "trailing_rv": trailing_rv})
    return panel.dropna()


# ----------------------------------------------------------------------------
# 2. The premium
# ----------------------------------------------------------------------------

def premium_stats(panel: pd.DataFrame) -> dict:
    """
    premium = iv - fwd_rv.

    Positive on average means the market persistently pays for more protection
    than turns out to be needed - compensation for holding the other side of
    that risk, which is why it's called a *premium* rather than a forecast error.
    """
    premium = panel.iv - panel.fwd_rv
    t_stat, p_value = stats.ttest_1samp(premium, 0)
    return {
        "premium": premium,
        "mean": premium.mean(),
        "median": premium.median(),
        "pct_positive": (premium > 0).mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "corr_iv_fwd": panel.iv.corr(panel.fwd_rv),
    }


# ----------------------------------------------------------------------------
# 3. Is the forecast any good?
# ----------------------------------------------------------------------------

def mincer_zarnowitz(panel: pd.DataFrame):
    """
    Regress realized volatility on implied volatility:

        fwd_rv = a + b * iv + error

    A perfect forecast would have a = 0 and b = 1 (a one-to-one mapping from
    forecast to outcome, no adjustment needed). This is a plain OLS regression;
    the name "Mincer-Zarnowitz" is just the standard reference for this
    forecast-evaluation setup in the finance literature.
    """
    X = sm.add_constant(panel.iv)
    return sm.OLS(panel.fwd_rv, X).fit()


def encompassing_regression(panel: pd.DataFrame):
    """
    Multiple regression testing whether one forecast subsumes another:

        fwd_rv = a + b1*iv + b2*trailing_rv + error

    If b2 is statistically indistinguishable from zero, the implied volatility
    forecast has already captured whatever information the trailing measure
    could provide.
    """
    X = sm.add_constant(panel[["iv", "trailing_rv"]])
    return sm.OLS(panel.fwd_rv, X).fit()


def forecast_accuracy(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Score each forecast against what actually happened, two ways:

      RMSE (root mean squared error) - squares errors before averaging, so a
      few huge misses dominate the score. Rewards a forecast that avoids being
      badly wrong, even if it's frequently a little off.

      MAE (mean absolute error) - averages the size of errors directly, so
      many small misses matter as much as a few large ones.

    A forecast can win on one and lose on the other, and that split is itself
    informative about *how* a forecast is wrong, not just whether it's wrong.
    """
    err_iv = panel.fwd_rv - panel.iv
    err_naive = panel.fwd_rv - panel.trailing_rv
    return pd.DataFrame({
        "RMSE": [np.sqrt((err_iv**2).mean()), np.sqrt((err_naive**2).mean())],
        "MAE": [err_iv.abs().mean(), err_naive.abs().mean()],
    }, index=["Implied volatility (VIX)", "Trailing realized volatility"])


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


def plot_timeseries(panel: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(panel.index, panel.iv, color=ACCENT, lw=1.1, label="Implied volatility (VIX)")
    ax.plot(panel.index, panel.fwd_rv, color=ACCENT_2, lw=0.9, alpha=0.85,
            label="Volatility that actually followed")
    ax.set_ylabel("Annualized volatility (%)", fontsize=9.5, color=INK)
    ax.set_title("What the Market Expected vs. What Happened",
                 fontsize=13, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_calibration(panel: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(panel.iv, panel.fwd_rv, s=6, color=ACCENT, alpha=0.18, edgecolors="none")
    lo, hi = 0, max(panel.iv.max(), panel.fwd_rv.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_2, lw=1.2, label="Perfect forecast")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Implied volatility on the day (%)", fontsize=9.5, color=INK)
    ax.set_ylabel("Volatility realized over the next month (%)", fontsize=9.5, color=INK)
    ax.set_title("Calibration: IV vs. What Followed", fontsize=12.5, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distribution(stats_dict: dict, path: str):
    premium = stats_dict["premium"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(premium, bins=80, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(stats_dict["mean"], color=ACCENT_2, ls="--", lw=1.3,
               label=f"Mean premium: {stats_dict['mean']:.1f} points")
    ax.set_xlabel("Implied volatility minus what followed (percentage points)",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("Number of trading days", fontsize=9.5, color=INK)
    ax.set_title("The Volatility Risk Premium, Day by Day",
                 fontsize=12.5, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Run
# ----------------------------------------------------------------------------

def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Downloading SPY and VIX history...")
    df = load_data()
    panel = build_panel(df)
    print(f"  {len(panel):,} trading days, {panel.index.min():%Y-%m-%d} to {panel.index.max():%Y-%m-%d}\n")

    print("Measuring the volatility risk premium...")
    prem = premium_stats(panel)
    print(f"  Correlation, IV vs. subsequent realized vol: {prem['corr_iv_fwd']:.3f}")
    print(f"  Mean premium:            {prem['mean']:.2f} points")
    print(f"  Share of days IV > realized: {100*prem['pct_positive']:.1f}%")
    print(f"  t-statistic: {prem['t_stat']:.1f}  (p = {prem['p_value']:.2e})\n")

    print("Testing whether IV is an unbiased forecast (Mincer-Zarnowitz)...")
    mz = mincer_zarnowitz(panel)
    print(f"  intercept = {mz.params['const']:.2f}   slope on IV = {mz.params['iv']:.3f}")
    print(f"  (unbiased forecast would have intercept 0, slope 1)\n")

    print("Testing whether IV subsumes trailing realized volatility...")
    enc = encompassing_regression(panel)
    print(f"  IV coefficient:          {enc.params['iv']:.3f}  (p = {enc.pvalues['iv']:.4f})")
    print(f"  Trailing RV coefficient: {enc.params['trailing_rv']:.3f}  (p = {enc.pvalues['trailing_rv']:.4f})\n")

    acc = forecast_accuracy(panel)
    print("Forecast accuracy (lower is better):")
    print(acc.to_string(float_format=lambda v: f"{v:6.2f}"))
    print()

    plot_timeseries(panel, f"{CHART_DIR}/vrp_timeseries.png")
    plot_calibration(panel, f"{CHART_DIR}/vrp_calibration.png")
    plot_distribution(prem, f"{CHART_DIR}/vrp_distribution.png")

    with open(f"{RESULT_DIR}/vrp_summary.txt", "w") as f:
        f.write("MINCER-ZARNOWITZ: fwd_rv = a + b*iv\n")
        f.write("=" * 70 + "\n")
        f.write(str(mz.summary()))
        f.write("\n\nENCOMPASSING: fwd_rv = a + b1*iv + b2*trailing_rv\n")
        f.write("=" * 70 + "\n")
        f.write(str(enc.summary()))
        f.write("\n\nFORECAST ACCURACY\n")
        f.write("=" * 70 + "\n")
        f.write(acc.to_string())

    print("Wrote charts/ and results/vrp_summary.txt")
    return panel, prem, mz, enc, acc


if __name__ == "__main__":
    main()
