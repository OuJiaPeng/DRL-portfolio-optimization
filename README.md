💡 RL-Markowitz Portfolio Optimization
This project compares two distinct portfolio optimization approaches: Markowitz Portfolio Optimization and Reinforcement Learning (RL) Portfolio Optimization using PPO.

✨ Features

Classical Markowitz Optimization with Efficient Frontier and Sharpe Ratio Maximization

Deep Reinforcement Learning (PPO) Agent for Dynamic Portfolio Allocation

Performance Comparison between Naive, Markowitz, and RL Portfolios

Visualizations for Portfolio Weights and Wealth Growth

🗂 Project Structure

RL-Markowitz-portfolio-optimization/
├── RL_PPO/                              # Reinforcement Learning (PPO) Module
│   ├── RL_portfolio.py                  # Training Script
│   ├── evaluate_RL_portfolio.py         # Evaluation Script
│   ├── RL_average_weights_pie.png       # Results (Visualizations)
│   ├── RL_weights_over_time.png
│   ├── RL_wealth_growth.png
│   └── ppo_portfolio_model_sharpe.zip   # Saved Model
│
├── markowitz/                           # Markowitz Optimization Module
│   ├── markowitz_portfolio.ipynb        # Jupyter Notebook for Markowitz
│   ├── markowitz_allocations_pie.png    # Results (Visualizations)
│   ├── markowitz_evaluation_metrics.txt # Metrics (Text Output)
│   └── markowitz_wealth_growth.png
│
├── naive/                               # Naive (Baseline) Portfolio
│   └── ETFcloseprices.png               # Data/Visualization for Naive Portfolio
│
├── README.md                            # Project Overview and Guide
├── requirements.txt                     # Dependency List
└── .gitignore                           # Git Ignore File

🛠️ Setup

Clone this repository:

git clone https://github.com/OuJiaPeng/RL-Markowitz-portfolio-optimization.git
cd RL-Markowitz-portfolio-optimization

Install the required packages:

pip install -r requirements.txt

🚀 Usage

Run the Markowitz optimization (Jupyter Notebook):

jupyter notebook markowitz/markowitz_portfolio.ipynb

Train the RL agent:

python RL_PPO/RL_portfolio.py

Evaluate the RL agent:

python RL_PPO/evaluate_RL_portfolio.py

📊 Results

Comparative analysis of Naive, Markowitz, and RL portfolios.

Key metrics: Return, Volatility, Sharpe Ratio.

📝 Future Improvements

Add transaction costs and turnover penalties

Optimize PPO hyperparameters

Explore advanced models (LSTM, Transformer) for RL.

