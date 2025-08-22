## RL and Markowitz Portfolio Optimization
This project compares Markowitz and DRL (Deep Reinforcement Learning) based portfolio optimization.

More details can be found in the **"NEW_project_slides.pdf"**

(More formal but outdated in **"OLD_project_slides.pdf"**)

---

## Data & Test Window
Test window: 2025-01-02 to 2025-07-01 (6 months of out-of-sample trading days).  
Universe (10 liquid ETFs spanning equities, intl, small-cap, EM, bonds, real assets & commodities): `SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, GLD, USO`.

## Method Summary
| Approach | Brief Description | Rebalance | Lookback / Refit | Objective |
|----------|------------------|-----------|------------------|-----------|
| Naive Equal Weight | Static 10% per ETF | None after start | N/A | Baseline diversification |
| Rolling Markowitz | Mean-Variance (expected excess return vs variance) using rolling window | Daily | 6-month rolling window | Max Sharpe (risk-free 4% annual) |
| RL (PPO) | Policy learns allocation weights on a simplex; state includes recent prices, returns & indicators | Daily (policy action); policy retrained monthly | Monthly refit on expanding data | Maximize risk-adjusted growth  |

## Results (Out-of-Sample 2025-01-01 → 2025-07-01)

Core risk/return comparison:

| Portfolio | Annualized Return | Volatility | Sharpe | Max Drawdown | CAGR | Notes |
|-----------|------------------:|-----------:|-------:|-------------:|-----:|-------|
| Naive (Equal Wt) | 12.67% | 15.44% | 0.82 | -12.05% | 16.48% | Broad diversified baseline |
| Markowitz | 31.68% | 18.06% | 1.75 | -5.80% | 40.56% | Highest raw & excess return |
| RL (PPO Agent) | 28.63% | 12.31% | 2.00 | -6.35% | 32.13% | Best risk-adjusted (Sharpe 2.0) |

Additional RL concentration / efficiency metrics:

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Cumulative Return | 13.06% | Raw growth over ~0.44 yrs (annualized 28.63%) |
| Calmar Ratio | 5.06 | Strong return per unit MDD |
| HHI (Concentration) | 0.437 | Moderate concentration (1 = single asset) |
| Effective Holdings | 2.3 | Equivalent number of equally weighted assets (due to adaptive concentration) |
| Weight Entropy | 1.016 | Lower entropy vs equal-weight (≈2.30 effective vs 10) |
| Max Single Weight | 54.0% | Policy selectively concentrates in strongest regime assets |

### Key Takeaways
1. Risk Efficiency: RL achieves similar annualized return to Markowitz with materially lower volatility (↓ ~32%), lifting Sharpe from 1.75 → 2.00.
2. Return Leadership: Markowitz edges RL on raw (excess) return, but with higher variance; RL dominates on risk-adjusted basis.
3. Drawdown Control: Both optimized approaches materially reduce max drawdown vs naive; RL maintains competitive drawdown despite selective concentration.
4. Adaptive Concentration: RL concentrates (Effective Holdings ≈ 2.3) when signal quality is high, otherwise diversifies—unlike static equal-weight or strictly optimization-based Markowitz which can overfit means/covariances.
5. Capital Efficiency: High Calmar (5.06) indicates strong return per unit downside risk, supportive for leveraged overlays (if desired) while keeping absolute drawdowns modest.

## Monte Carlo (MC) Benchmark
Purpose: Provide a distributional baseline of achievable risk-adjusted performance under uninformed daily random rebalancing (uniform draw on the 10-asset simplex each day) to contextualize systematic strategies.

Latest run (updated): 1,000,000 simulated random portfolios (daily re-sampled weights), same test window.

Tier summaries (means within top X% of Sharpe-ranked simulations):

| Tier | Count | Mean Sharpe | Mean CAGR | Mean Vol | Mean MDD | Mean Final Wealth |
|------|------:|------------:|----------:|---------:|---------:|------------------:|
| Top 0.01% | 100 | 2.032 | 0.403 | 0.153 | -0.094 | 1.1780 |
| Top 0.1%  | 1,000 | 1.833 | 0.365 | 0.154 | -0.098 | 1.1625 |
| Top 1%    | 10,000 | 1.610 | 0.321 | 0.156 | -0.103 | 1.1444 |
| Top 50%   | 500,000 | 1.030 | 0.210 | 0.158 | -0.116 | 1.0967 |

Extremes across all 1,000,000 sims: Max Sharpe observed 2.465; Max CAGR 0.488.

Placement of strategies:
- RL PPO Sharpe 2.00 ≈ between the mean of the top 0.01% (2.03) and its lower tail; lies well inside the extreme right tail.
- Markowitz Sharpe 1.75 ≈ between mean of top 0.1% (1.83) and mean of top 1% (1.61); sits in roughly the top ~0.2–0.3% by interpolation.
- Equal-weight Sharpe 0.82 < mean of top 50% (1.03); near or below overall distribution midpoint.

Interpretation: Achieving Sharpe ≈2.0 under this regime-free random baseline is statistically rare (≈ top 0.01–0.02% region), strengthening the evidence that RL performance is not a random allocation artifact. Markowitz also occupies a high, but less extreme, tail percentile.

Stability note: 0.01% tier (n=100) now has sufficient sample size for a rough mean (relative SE ~10%); deeper tail claims (e.g., exact percentile of a single Sharpe) would need either more simulations or extreme value modeling.

## Metric Definitions (Concise)
| Metric | Definition (Simplified) |
|--------|-------------------------|
| Annualized Return | Geometric scaling of cumulative return to 1-year equivalent |
| Volatility | Std dev of daily returns × √252 |
| Sharpe | (Annualized Return − Annual RF) / Volatility (excess form) |
| Max Drawdown | Worst peak-to-trough portfolio equity decline |
| Calmar | Annualized Return / |Max Drawdown| |
| CAGR | (Ending Value / Start)^(1/Years) − 1 |
| HHI | Σ w_i^2 (concentration index) |
| Effective Holdings | 1 / HHI |
| Weight Entropy | − Σ w_i log w_i (higher = more diversified) |

## High-Level Reproduction Outline
1. Load & enrich ETF price data (indicators engineered in `data/`).
2. Train RL PPO policy on rolling / expanding window; save best model (`models/`).
3. Run out-of-sample evaluation (generates `results/` metrics & plots).
4. Run rolling Markowitz optimizer with same universe & risk-free assumption.
5. (Optional) Execute large-scale Monte Carlo to benchmark distribution tail.

## Next Potential Improvements
- Transaction cost & slippage modeling
- Regime detection layer feeding policy context vectors
- Ensemble of RL + robust covariance optimizer
- Risk parity / volatility targeting overlay for capital scaling

> Disclaimer: Historical simulation; no guarantee of future performance. Sharpe ratios over short horizons can regress materially.