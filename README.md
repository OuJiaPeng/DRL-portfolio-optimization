## DRL Portfolio Optimization
This project explores DRL (Deep Reinforcement Learning) based portfolio optimization with comparison to classical baselines.

Through many interviews and continuous refinement, I think I have reached a final product.

More details can be found in the **"UPDATED_project_slides.pdf"**

---

## Data & Test Window
Test window: 2025-01-02 to 2025-07-01 (6 months of out-of-sample trading days).  
Universe (10 ETFs; equities, intl, small-cap, EM, bonds, real assets, commodities): `SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, GLD, USO`.

## Method Summary
| Approach | Brief Description | Rebalance | Lookback / Refit |
|----------|------------------|-----------|------------------|
| RL (PPO) | PPO dynamically allocates portfolio weights to maximize a reward function | Daily | Monthly refit on expanding test data |
| Rolling Markowitz | Mean-Variance optimization using rolling window | Daily | 6-month rolling window |
| Naive Equal Weight | Static 10% per ETF | None after start | N/A |


## Results

Core risk/return comparison:

| Portfolio | Sharpe | Annualized Excess Return | Volatility | Max Drawdown | CAGR |
|-----------|-------:|------------------:|-----------:|-------------:|-----:|
| RL (PPO Agent) | 2.00 | 24.63% | 12.31% | -6.35% | 32.13% |
| Markowitz | 1.75 | 31.68% | 18.06% | -5.80% | 40.56% |
| Naive (Equal Wt) | 0.82 | 12.67% | 15.44% | -12.05% | 16.48% |
| SPY (Long) | 0.06 | 1.49% | 26.48% | -19.00% | 0.78% |


### Key Takeaways
1. **Risk Efficiency:** RL attains 22.3\% lower excess return but also 31.8\% lower vol vs Markowitz, lifting Sharpe from 1.75 → 2.00.
2. **Adaptive Concentration:** RL can concentrate when signal quality is high, otherwise diversifies. This is unlike equal-weight or optimization-based Markowitz which can overfit means/covariances.
3. **Monte Carlo tail positioning (1,000,000 sims):** Markowitz Sharpe 1.75 ≈ top 0.3%; RL Sharpe 2.00 ≈ top 0.01% of simulated random-allocation paths.  

