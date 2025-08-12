## RL and Markowitz Portfolio Optimization
This project compares Markowitz and DRL (Deep Reinforcement Learning) based portfolio optimization.

More details can be found in the **"NEW_project_slides.pdf"**
(More formal but outdated in **"OLD_project_slides.pdf"**)

---

## Results (From 2025-01-01 to 2025-07-01)

| Portfolio         | Annualized Return | Volatility | Sharpe Ratio |
|------------------|-------------------|------------|--------------|
| Naive (Equal Wt) | 12.67%            | 15.44%     | 0.82         |
| Markowitz        | 31.97%            | 17.47%     | 1.83         |
| RL (PPO Agent)   | 18.99%            | 10.52%     | 1.81         |

## MC Simulation

Daily random rebalance; 1,000,000 Simulation Runs

Top 0.01%: >2.05 SR (RL here)
Top 0.1%: >1.84 SR (Markowitz here)