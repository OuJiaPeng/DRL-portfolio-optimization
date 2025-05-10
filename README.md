💡 RL-Markowitz Portfolio Optimization

    This project compares Markowitz and Reinforcement Learning based portfolio optimization.

    More details can be found in the "Project_slides.pdf"

✨ Features

    Classical Markowitz Optimization; aimed to maximize Sharpe Ratio

    Deep Reinforcement Learning (PPO) Agent for Dynamic Portfolio Allocation, reward is Sharpe based

    Performance Comparison between Naive, Markowitz, and RL Portfolios

    Visualizations for Portfolio Weights and Wealth Growth

🗂 Project Structure

    RL-Markowitz-portfolio-optimization/  
    ├── RL_PPO/                              # Reinforcement Learning (PPO) Module  
    │   ├── RL_portfolio.py                  # Training Script  
    │   ├── evaluate_RL_portfolio.py         # Evaluation Script  
    │   ├── RL_average_weights_pie.png       # Results (Visuals & Metrics)  
    │   ├── RL_weights_over_time.png  
    │   ├── RL_wealth_growth.png  
    │   ├── RL_evaluation_summary.txt  
    │   └── ppo_portfolio_model_sharpe.zip   # Saved Model  
    │
    ├── markowitz/                           # Markowitz Optimization Module  
    │   ├── markowitz_portfolio.ipynb        # Jupyter Notebook for Markowitz  
    │   ├── markowitz_allocations_pie.png    # Results (Visuals & Metrics)  
    │   ├── markowitz_wealth_growth.png  
    │   └── markowitz_evaluation_metrics.txt 
    │  
    ├── naive/                               # Naive (Baseline) Portfolio  
    │   ├── naive_portfolio_plot.png         # Results (Visuals & Metrics)  
    │   └── naive_metrics.txt         
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

🚀 How to use

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

