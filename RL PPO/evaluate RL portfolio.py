# =============================
# RL Portfolio Evaluation Script (Sharpe-Optimized)
# =============================
 
import os

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ========== 1. Load Test Data ==========

etfs = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'GLD', 'USO']
print("Loading test data...")
prices = yf.download(etfs, start="2018-01-01", end="2024-01-01", group_by='ticker', auto_adjust=True, progress=False)
adj_close_prices = prices.xs('Close', level=1, axis=1).dropna()
test_prices = adj_close_prices.loc['2023-01-01':]
test_returns = test_prices.pct_change().dropna()

# ========== 2. Define Matching Environment ==========

class PortfolioEnv(gym.Env):
    def __init__(self, returns, window_size=30, risk_aversion_coef=1.0):
        super().__init__()
        self.returns = returns.values
        self.window_size = window_size
        self.n_assets = self.returns.shape[1]
        self.risk_aversion_coef = risk_aversion_coef
        self.portfolio_return_buffer = []

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.n_assets), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.portfolio_return_buffer = []
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        return self.returns[self.current_step - self.window_size:self.current_step, :]

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        else:
            action /= np.sum(action)

        portfolio_return = np.dot(self.returns[self.current_step], action)
        log_return = np.log(1 + portfolio_return)

        self.portfolio_return_buffer.append(portfolio_return)
        if len(self.portfolio_return_buffer) >= 5:
            vol_penalty = np.std(self.portfolio_return_buffer[-5:])
        else:
            vol_penalty = 0.01

        reward = log_return - self.risk_aversion_coef * vol_penalty

        self.current_step += 1
        self.done = self.current_step >= len(self.returns) - 1

        return self._get_observation(), reward, self.done, False, {
            "scaled_action": action,
            "raw_return": portfolio_return
        }

# ========== 3. Load Model ==========

model_path = os.path.join(os.path.dirname(__file__), "ppo_portfolio_model_sharpe")
model = PPO.load(model_path)

# ========== 4. Evaluation Loop ==========

n_runs = 10
sharpe_list = []
return_list = []
volatility_list = []

for run in range(n_runs):
    test_env = DummyVecEnv([lambda: PortfolioEnv(test_returns, window_size=30)])
    obs = test_env.reset()
    wealth = [1.0]
    daily_returns = []

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = test_env.step(action)
        r = info[0]["raw_return"]
        wealth.append(wealth[-1] * (1 + r))
        daily_returns.append(r)
        if done:
            break

    daily_returns = np.array(daily_returns)
    mean_ret = daily_returns.mean() * 252
    std_ret = daily_returns.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    sharpe_list.append(sharpe)
    return_list.append(mean_ret)
    volatility_list.append(std_ret)

# ========== 5. Save Metrics ==========

avg_return = np.mean(return_list)
std_return = np.std(return_list)
avg_volatility = np.mean(volatility_list)
std_volatility = np.std(volatility_list)
avg_sharpe = np.mean(sharpe_list)
std_sharpe = np.std(sharpe_list)

print(f"Average Sharpe Ratio: {avg_sharpe:.2f} ± {std_sharpe:.2f}\n")

with open(os.path.join(os.path.dirname(__file__), "RL_evaluation_summary.txt"), "w") as f:
    f.write(f"Average Annualized Return: {avg_return:.2%} ± {std_return:.2%}\n")
    f.write(f"Average Annualized Volatility: {avg_volatility:.2%} ± {std_volatility:.2%}\n")
    f.write(f"Average Sharpe Ratio: {avg_sharpe:.2f} ± {std_sharpe:.2f}\n")

print("Evaluation complete — summary saved to RL_evaluation_summary.txt")


# ========== 6. Single Final Run for Plots ==========

test_env = DummyVecEnv([lambda: PortfolioEnv(test_returns, window_size=30)])
obs = test_env.reset()

wealth = [1.0]
daily_returns = []
actions_taken = []

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    actions_taken.append(action[0])
    wealth.append(wealth[-1] * (1 + info[0]["raw_return"]))
    daily_returns.append(info[0]["raw_return"])
    if done:
        break

# ========== Plot 1: Wealth Growth ==========

plt.figure(figsize=(12, 6))
plt.plot(wealth)
plt.title('Wealth Growth in 2023 (RL Agent)')
plt.xlabel('Days')
plt.ylabel('Wealth')
plt.grid()
plt.savefig(os.path.join(os.path.dirname(__file__), 'RL_wealth_growth.png'))
plt.close()

# ========== Plot 2: Portfolio Weights Over Time ==========

actions_df = pd.DataFrame(actions_taken, columns=etfs)
plt.figure(figsize=(14, 6))
actions_df.plot(linewidth=1)
plt.title('RL Agent Portfolio Weights Over Time (2023)')
plt.xlabel('Days')
plt.ylabel('Weight')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),'RL_weights_over_time.png'))
plt.close()

# ========== Plot 3: Average Weights Pie Chart ==========

avg_weights = actions_df.mean()
plt.figure(figsize=(6, 6))
plt.pie(avg_weights, labels=avg_weights.index, autopct='%1.1f%%', startangle=140)
plt.title('Average Portfolio Weights (2023)')
plt.axis('equal')
plt.savefig(os.path.join(os.path.dirname(__file__),'RL_average_weights_pie.png'))
plt.close()
