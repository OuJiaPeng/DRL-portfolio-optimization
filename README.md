## RL can achieve SR > 2.0, experimenting with different policies to try and surpass

## RL and Markowitz Portfolio Optimization
This project compares Markowitz and DRL (Deep Reinforcement Learning) based portfolio optimization.

More details can be found in the **"NEW_project_slides.pdf"**

(More formal but outdated in **"OLD_project_slides.pdf"**)

---

## Data & Test Window
Test window: 2025-01-02 to 2025-07-01 (≈ 0.5 years of out-of-sample trading days).  
Universe (10 liquid ETFs spanning equities, intl, small-cap, EM, bonds, real assets & commodities): `SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, GLD, USO`.

## Method Summary
| Approach | Brief Description | Rebalance | Lookback / Refit | Objective |
|----------|------------------|-----------|------------------|-----------|
| Naive Equal Weight | Static 10% per ETF | None after start | N/A | Baseline diversification |
| Rolling Markowitz | Mean-Variance (expected excess return vs variance) using rolling window | Daily | 6-month rolling window | Max Sharpe (risk-free 4% annual) |
| RL (PPO) | Policy learns allocation weights on a simplex; state includes recent prices, returns & indicators | Daily (policy action); policy retrained monthly | Monthly refit on expanding data | Maximize risk-adjusted growth (implicit via reward = portfolio return - risk penalty) |

## Results (Out-of-Sample 2025-01-02 → 2025-07-01)

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
Purpose: Provide a distributional baseline of achievable Sharpe ratios under daily random rebalancing (uniform weights on the 10-asset simplex) to contextualize systematic strategies.

Setup:
- 1,000,000 simulated random weight paths
- Daily re-sampled portfolio weights (no transaction costs assumed)
- Same asset return stream & test window as above

Percentile positioning:
| Percentile Threshold | Sharpe Cutoff | Strategy Placement |
|----------------------|--------------:|-------------------|
| Top 0.01% | > 2.00 | RL sits here |
| Top 0.1% | > 1.84 | Markowitz sits here |
| Median (not shown) | (<< 1.0) | Random baseline much lower |

Interpretation: Achieving Sharpe ≥ 2.0 places the RL policy in the extreme right tail of the unconditional random allocation distribution, suggesting genuine signal extraction / dynamic risk management rather than luck.

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