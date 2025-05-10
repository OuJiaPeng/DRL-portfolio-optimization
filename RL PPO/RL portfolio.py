# =============================
# RL Portfolio Project (Sharpe-Oriented with Early Stopping)
# =============================

import os

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback


# ========== Custom PPO Policy ==========

class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                         activation_fn=nn.ReLU)


# ========== Custom Environment ==========

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


# ========== Sharpe Early Stopping  ==========

class SharpeEarlyStoppingCallback(BaseCallback):
    def __init__(self, val_env, patience=3, eval_freq=20000, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.num_bad_epochs = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.val_env.reset()
            returns = []
            wealth = [1.0]

            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.val_env.step(action)
                r = info["raw_return"]
                returns.append(r)
                wealth.append(wealth[-1] * (1 + r))
                if done:
                    break

            returns = np.array(returns)
            mean_ret = returns.mean() * 252
            std_ret = returns.std() * np.sqrt(252)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0

            if self.verbose:
                print(f"Eval Sharpe at step {self.n_calls}: {sharpe:.4f}")

            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                print("Early stopping triggered.")
                return False

        return True


# ========== Main Execution ==========

if __name__ == '__main__':
    import multiprocessing
    import yfinance as yf
    import matplotlib.pyplot as plt

    multiprocessing.freeze_support()

    # === [1] Download Data Once ===

    print("Downloading data...")
    etfs = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'GLD', 'USO']
    prices = yf.download(etfs, start="2018-01-01", end="2024-01-01", group_by='ticker', auto_adjust=True, progress=False)
    adj_close_prices = prices.xs('Close', level=1, axis=1).dropna()

    # === [2] Split Data ===

    train_prices = adj_close_prices.loc[:'2021-12-31']
    val_prices = adj_close_prices.loc['2022-01-01':'2022-12-31']
    test_prices = adj_close_prices.loc['2023-01-01':]

    train_returns = train_prices.pct_change().dropna()
    val_returns = val_prices.pct_change().dropna()
    test_returns = test_prices.pct_change().dropna()

    print("Training Period:", train_returns.index.min(), "to", train_returns.index.max())
    print("Validation Period:", val_returns.index.min(), "to", val_returns.index.max())
    print("Testing Period:", test_returns.index.min(), "to", test_returns.index.max())

    # === [3] Setup VecEnv and Validation Env ===

    def make_env():
        return lambda: PortfolioEnv(train_returns, window_size=30, risk_aversion_coef=1.0)

    n_envs = 4
    train_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    val_env = PortfolioEnv(val_returns, window_size=30, risk_aversion_coef=1.0)

    # === [4] Train PPO with Early Stopping and ProgressBar ===
    
    model = PPO(
        CustomMLPPolicy,
        train_env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        learning_rate=3e-4
    )

    callbacks = [
        SharpeEarlyStoppingCallback(val_env, patience=3, eval_freq=20000, verbose=1),
        ProgressBarCallback()
    ]

    model.learn(total_timesteps=1_000_000, callback=callbacks)

    model_path = os.path.join(os.path.dirname(__file__), "ppo_portfolio_model_sharpe")
    model.save(model_path)
    print(f"Training complete. Model saved as '{model_path}.zip'")
