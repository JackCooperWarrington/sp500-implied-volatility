"""
Robustness Checks
=================

The main paper reports pooled results across the full 2010-2026 sample. This
module tests whether those results are stable across subsamples, or whether
they are driven by particular periods.

Two checks:

  1. VOLATILITY RISK PREMIUM: pre-2020 vs. post-2020
     Tests whether the ~4-point gap between implied and realized volatility
     persists in both halves of the sample.

  2. PRE-FOMC DRIFT: pre-2020 vs. post-2020
     Tests whether the significant t-2 rise before FOMC meetings shows up in
     both subsamples.

The pre/post-2020 split is chosen because 2020 marks a structural break in
monetary policy (zero-rate policy, then rapid tightening) and in volatility
regime (COVID-19 volatility episode). If a finding survives across both
periods, that is meaningful evidence it is not driven by one episode.

Usage:  python robustness.py
Output: printed subsample statistics; used in Section 5.4 of the paper.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy import stats

from fomc_event_study import FOMC_DATES_RAW

warnings.filterwarnings("ignore")

START = "2010-01-01"
SPLIT_DATE = pd.Timestamp("2020-01-01")


def load_data():
    spy = yf.download("SPY", start=START, progress=False, auto_adjust=True)["Close"]
    vix = yf.download("^VIX", start=START, progress=False, auto_adjust=True)["Close"]
    spy.columns, vix.columns = ["SPY"], ["VIX"]
    return spy.join(vix, how="inner").dropna()


def build_vrp_panel(df):
    log_ret = np.log(df.SPY / df.SPY.shift(1))
    ann = np.sqrt(252) * 100
    fwd_rv = log_ret.shift(-21).rolling(21).std() * ann
    fwd_rv = fwd_rv.shift(-20)
    return pd.DataFrame({"iv": df.VIX, "fwd_rv": fwd_rv}).dropna()


def vrp_subsample_test(panel):
    """Volatility risk premium in pre-2020 vs. post-2020 subsamples."""
    results = []
    for label, mask in [
        ("2010-2019", panel.index < SPLIT_DATE),
        ("2020-2026", panel.index >= SPLIT_DATE),
    ]:
        sub = panel[mask]
        prem = sub.iv - sub.fwd_rv
        t, p = stats.ttest_1samp(prem, 0)

        X = sm.add_constant(sub.iv)
        mz = sm.OLS(sub.fwd_rv, X).fit()

        results.append({
            "period": label,
            "n_days": len(sub),
            "premium_mean": prem.mean(),
            "premium_t": t,
            "corr": sub.iv.corr(sub.fwd_rv),
            "slope": mz.params["iv"],
            "slope_se": mz.bse["iv"],
            "intercept": mz.params["const"],
        })
    return pd.DataFrame(results)


def fomc_subsample_test(vix):
    """Pre-FOMC drift at t-2 in pre-2020 vs. post-2020 subsamples."""
    fomc = pd.to_datetime([
        s.strip() for s in FOMC_DATES_RAW.replace("\n", ",").split(",") if s.strip()
    ])
    fomc = fomc[fomc.isin(vix.index)]
    dvix = vix.diff()

    results = []
    for label, mask in [
        ("2010-2019", fomc < SPLIT_DATE),
        ("2020-2026", fomc >= SPLIT_DATE),
    ]:
        sub_fomc = fomc[mask]
        tminus2 = []
        for date in sub_fomc:
            idx = vix.index.get_loc(date)
            if idx - 2 >= 0:
                tminus2.append(dvix.iloc[idx - 2])
        tminus2 = pd.Series(tminus2).dropna()
        t, p = stats.ttest_1samp(tminus2, 0)

        results.append({
            "period": label,
            "n_meetings": len(sub_fomc),
            "t_minus_2_mean": tminus2.mean(),
            "t_stat": t,
            "p_value": p,
        })
    return pd.DataFrame(results)


def main():
    print("Loading data...")
    df = load_data()
    vix = df.VIX
    panel = build_vrp_panel(df)

    print("\nVolatility risk premium: subsample stability")
    print("=" * 66)
    vrp = vrp_subsample_test(panel)
    print(vrp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nPre-FOMC drift at t-2: subsample stability")
    print("=" * 66)
    fomc = fomc_subsample_test(vix)
    print(fomc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    return vrp, fomc


if __name__ == "__main__":
    main()
