## RL and Markowitz Portfolio Optimization
This project compares Markowitz and Reinforcement Learning based portfolio optimization.

More details can be found in the **"Project_slides.pdf"**

Overall, there are many aspects of this which could be improved on. Working towards a better solution [here](https://github.com/OuJiaPeng/Portfolio-Optimization).

- Need to consider drawdown, turnover, and slippage
- Model could use a lot more complexity and factor input
- Just a handful of issues, will be fixed in new project.

---

## Results

| Portfolio         | Annualized Return | Volatility | Sharpe Ratio |
|------------------|-------------------|------------|--------------|
| Naive (Equal Wt) | 15.00%            | 10.70%     | 1.40         |
| Markowitz        | 25.37%            | 10.76%     | 2.36         |
| RL (PPO Agent)   | 18.99%            | 10.52%     | 1.81         |

Very surprised Markowitz performed this well — it feels like doing the "wrong" steps and getting the right answer on a test.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
```bash
python src/training/train_only.py --mode HYBRID_ACTION --strength 0.8 --timesteps 300000
```

### 3. Evaluate Model
```bash
python src/evaluation/evaluate_only.py --model-path models/saved/your_model.zip
```

### 4. Analyze Results
Open `notebooks/markowitz_analysis.ipynb` for detailed performance analysis.

---

## Features
- **Classical Markowitz Optimization:** aimed to maximize Sharpe Ratio  
- **Deep Reinforcement Learning (PPO) Agent:** dynamic Portfolio Allocation, reward is Sharpe based  
- **Performance Comparison:** Markowitz, RL, and a Naive (equal weights) portfolio  
- **Visualizations:** Portfolio Weights and Wealth Growth  

---

## Project Structure

    rl_markowitz_project/
    ├── src/                                   # Main Source Code (Modular Architecture)
    │   ├── envs/                             # RL Environments
    │   │   ├── markowitz_guided_env.py       # Markowitz-guided environment
    │   │   └── optimized_portfolio_env.py    # Optimized base environment
    │   ├── training/                         # Training Scripts
    │   │   ├── train_only.py                 # Standalone training
    │   │   └── training_utils.py             # Training utilities
    │   ├── evaluation/                       # Evaluation Scripts
    │   │   ├── evaluate_only.py              # Standalone evaluation
    │   │   └── evaluation_utils.py           # Evaluation utilities
    │   ├── utilities/                        # Core Utilities
    │   │   ├── load_minimal_data.py          # Optimized data loader
    │   │   └── markowitz_weights.py          # Markowitz optimization
    │   └── config.py                         # Centralized configuration
    ├── models/saved/                         # Trained Models
    │   └── simple_mlp_policy.py              # Neural network architecture
    ├── data/processed/                       # Processed Data
    │   └── etf_data_with_indicators.csv      # Main dataset
    ├── results/outputs/                      # Results & Analysis
    ├── notebooks/                            # Jupyter Analysis
    │   ├── markowitz_analysis.ipynb          # Classical optimization
    │   └── naive_baseline.ipynb              # Equal-weight baseline
    ├── requirements.txt                      # Dependencies
    └── README.md                             # This file

---

## Future Improvements

- Add transaction costs and turnover penalties

- Better optimize PPO hyperparameters

- Incorporate complexities for RL model

---

