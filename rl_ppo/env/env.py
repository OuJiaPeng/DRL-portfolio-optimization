import os
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl_ppo.config import Config


class PortfolioEnv(gym.Env):

    """
    Reward components (applied in order):
      base = portfolio (excess) return - turnover_cost
      + movement bonus
      + momentum bonus (optional)
      - variance penalty (optional)
      + two-sided HHI band shaping (anti-uniform & anti-concentration)
      + advantage (relative-return tilt)
      - logit L2 (optional)
      => optional reward normalization (rolling std/mean)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, features: np.ndarray, prices: np.ndarray, config_overrides: dict = None):
        super().__init__()
        assert features.ndim == 2 and prices.ndim == 2, "Features and prices must be 2D"
        assert features.shape[0] == prices.shape[0], "Features/prices time dimension mismatch"
        self.features = features.astype(np.float32)

        # Store config overrides for refit-specific parameters
        self.config_overrides = config_overrides or {}

        self.prices = prices.astype(np.float32)
        self.n_assets = self.prices.shape[1]
        assert self.n_assets == len(Config.ETF_TICKERS), "Price columns != ETF_TICKERS length"

        self.action_space = spaces.Box(low=-10, high=10, shape=(self.n_assets,), dtype=np.float32)
        self.obs_dim = self.features.shape[1] + (self.n_assets if getattr(Config, 'INCLUDE_PREV_WEIGHTS', True) else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # State
        self._rng = np.random.default_rng(int(getattr(Config, 'SEED', 0)))
        self.t = 0
        self.w_prev = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._rew_hist: list[float] = []
        self._asset_bias: np.ndarray | None = None

        # Debug attributes (inactive)
        self.debug = False
        self.debug_limit = 0
        self.debug_path = ''
        self._debug_rows = 0

    def _get_config_value(self, attr_name: str, default=None):
        """Get config value with override support for refit parameters."""
        if attr_name in self.config_overrides:
            return self.config_overrides[attr_name]
        return getattr(Config, attr_name, default)

    def step(self, action):  # type: ignore[override]
        a = np.asarray(action, dtype=np.float32)
        # Logit clipping & temperature
        logit_clip = float(getattr(Config, 'LOGIT_CLIP', 0.0) or 0.0)
        if logit_clip > 0.0:
            a = np.clip(a, -logit_clip, logit_clip)
        temp = float(getattr(Config, 'ACTION_TEMPERATURE', 1.0) or 1.0)
        if temp <= 0:
            temp = 1.0
        # Optional per-episode asset bias to break symmetry
        if getattr(Config, 'USE_ASSET_BIAS', False):
            if self._asset_bias is None:
                # should have been set at reset, fallback here
                std = float(getattr(Config, 'ASSET_BIAS_STD', 0.0) or 0.0)
                if std > 0:
                    self._asset_bias = self._rng.normal(0.0, std, size=self.n_assets).astype(np.float32)
                else:
                    self._asset_bias = np.zeros(self.n_assets, dtype=np.float32)
            a = a + self._asset_bias

        z = (a / temp) - (a / temp).max()
        w = np.exp(z); w = w / (w.sum() + 1e-9)

        # Optional caps
        max_w = float(self._get_config_value('MAX_POSITION_SIZE', 1.0) or 1.0)
        min_w = float(self._get_config_value('MIN_POSITION_SIZE', 0.0) or 0.0)
        if max_w < 1.0 or min_w > 0.0:
            w = np.clip(w, min_w, max_w)
            s = w.sum()
            w = (w / s) if s > 1e-9 else np.ones_like(w) / len(w)

        # Rebalance frequency
        k_reb = int(self._get_config_value('REBALANCE_FREQ', 1) or 1)
        w_eff = self.w_prev if (k_reb > 1 and (self.t % k_reb != 0)) else w

        # Returns
        p0 = self.prices[self.t]
        p1 = self.prices[self.t + 1]
        ret_vec = (p1 - p0) / (p0 + 1e-9)
        port_ret = float(np.dot(w_eff, ret_vec))

        # Base reward
        rf = float(getattr(Config, 'RISK_FREE_DAILY', 0.0)) if getattr(Config, 'REWARD_USE_EXCESS_RET', False) else 0.0
        cost_bps = float(self._get_config_value('TURNOVER_COST', 0.00025))
        turnover_cost = 0.0 if self._get_config_value('DISABLE_TURNOVER_COST', False) else float(np.sum(np.abs(w_eff - self.w_prev)) * cost_bps)
        reward = (port_ret - rf) - turnover_cost

        # Movement
        move_coef = float(getattr(Config, 'MOVE_BONUS_COEF', 0.0) or 0.0)
        if move_coef > 0.0:
            reward += move_coef * float(np.sum(np.abs(w_eff - self.w_prev)))

        # Momentum
        alpha_m = float(getattr(Config, 'MOMENTUM_ALPHA', 0.0) or 0.0)
        if alpha_m != 0.0:
            k = int(getattr(Config, 'MOMENTUM_LOOKBACK', 20) or 20)
            if self.t >= k:
                window_p0 = self.prices[self.t - k]
                mom_vec = (p0 - window_p0) / (window_p0 + 1e-9)
                reward += alpha_m * float(np.dot(w_eff, mom_vec))

        # Variance penalty
        alpha_r = float(getattr(Config, 'RISK_PENALTY_ALPHA', 0.0) or 0.0)
        if alpha_r > 0.0:
            win = int(getattr(Config, 'RISK_PENALTY_WINDOW', 20) or 20)
            # Need at least `win` historical returns; returns window uses aligned pairs
            if self.t >= win:
                # Use last `win` daily returns ending at current p0 (exclude the pending p1 move)
                # Returns: P[t-win+1:t+1] / P[t-win:t] - 1  -> shape (win, n_assets)
                ret_hist = (self.prices[self.t - win + 1:self.t + 1] / self.prices[self.t - win:self.t] - 1.0)
                if ret_hist.shape[0] == win:  # defensive
                    cov = np.cov(ret_hist.T, ddof=0)
                    port_var = float(w_eff @ cov @ w_eff)
                    reward -= alpha_r * port_var

        # Diversification shaping
        scheme = getattr(Config, 'DIVERSITY_SCHEME', 'none')
        hhi_tmp = float(np.sum(w_eff ** 2))
        if scheme == 'two_sided_band':
            lower = float(getattr(Config, 'HHI_LOWER_BAND', 0.12))
            upper = float(getattr(Config, 'HHI_UPPER_BAND', 0.28))
            pen_low = float(getattr(Config, 'UNI_TOO_LOW_COEF', 0.0))
            pen_high = float(getattr(Config, 'CONC_PEN_COEF', 0.0))
            center_bonus = float(getattr(Config, 'BAND_CENTER_BONUS', 0.0))
            if hhi_tmp < lower:
                reward -= pen_low * (lower - hhi_tmp)
            elif hhi_tmp > upper:
                reward -= pen_high * (hhi_tmp - upper)
            else:
                reward += center_bonus
        elif scheme == 'linear':
            div_alpha = float(getattr(Config, 'DIVERSITY_BONUS_ALPHA', 0.0) or 0.0)
            if div_alpha != 0.0:
                reward += div_alpha * (1.0 - hhi_tmp)

        # ENH penalty
        enh_pen_coef = float(getattr(Config, 'ENH_PENALTY_COEF', 0.0) or 0.0)
        if enh_pen_coef > 0.0:
            enh = 1.0 / (hhi_tmp + 1e-9)
            min_enh = float(getattr(Config, 'MIN_ENH', 0.0) or 0.0)
            if min_enh > 0.0 and enh < min_enh:
                reward -= enh_pen_coef * (min_enh - enh)

        # Advantage bonus
        adv_coef = float(getattr(Config, 'ADV_COEF', 0.0) or 0.0)
        if adv_coef != 0.0:
            rel = ret_vec - ret_vec.mean()
            reward += adv_coef * float(np.dot(w_eff, rel))

        # Logit L2
        logit_l2 = float(getattr(Config, 'LOGIT_L2_PENALTY', 0.0) or 0.0)
        if logit_l2 > 0.0:
            reward -= logit_l2 * float(np.mean(a ** 2))

        # Scale
        reward *= float(getattr(Config, 'REWARD_SCALE', 1.0) or 1.0)

        # Reward normalization
        if getattr(Config, 'REWARD_NORMALIZE', False):
            self._rew_hist.append(reward)
            win = int(getattr(Config, 'REWARD_STD_WINDOW', 60) or 60)
            if len(self._rew_hist) > win:
                self._rew_hist = self._rew_hist[-win:]
            std = float(np.std(self._rew_hist, ddof=0))
            if std > 1e-8:
                if getattr(Config, 'REWARD_CENTER_MEAN', False):
                    mean_ = float(np.mean(self._rew_hist))
                    reward = (reward - mean_) / std
                else:
                    reward = reward / std

        # Logging
        turnover_realized = float(np.sum(np.abs(w_eff - self.w_prev)))
        self.w_prev = w_eff
        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        truncated = False
        obs = self._get_obs()
        hhi = float(np.sum(w_eff ** 2))
        enh = float(1.0 / (hhi + 1e-9))
        info: Dict[str, Any] = {
            'port_ret': port_ret,
            'turnover_cost': turnover_cost,
            'weights': w_eff.copy(),
            'hhi': hhi,
            'enh': enh,
            'turnover': turnover_realized,
            'raw_return': port_ret,
            'asset_bias': None if self._asset_bias is None else self._asset_bias.copy(),
        }

    # (Debug CSV logging removed for final release)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.t = 0
        self._rew_hist = []
        if getattr(Config, 'RANDOM_INIT_WEIGHTS', False):
            raw = self._rng.normal(size=self.n_assets)
            raw = raw - raw.mean()
            w = np.exp(raw); self.w_prev = (w / w.sum()).astype(np.float32)
        else:
            self.w_prev = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
            
        # Draw new per-episode asset bias
        if getattr(Config, 'USE_ASSET_BIAS', False):
            std = float(getattr(Config, 'ASSET_BIAS_STD', 0.0) or 0.0)
            if std > 0:
                self._asset_bias = self._rng.normal(0.0, std, size=self.n_assets).astype(np.float32)
            else:
                self._asset_bias = np.zeros(self.n_assets, dtype=np.float32)
        else:
            self._asset_bias = None
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        feat_t = self.features[self.t]
        if getattr(Config, 'INCLUDE_PREV_WEIGHTS', True):
            prev = self.w_prev if not getattr(Config, 'ZERO_PREV_WEIGHTS', False) else np.zeros_like(self.w_prev)
            return np.concatenate([feat_t, prev], axis=0).astype(np.float32)
        return feat_t.astype(np.float32)

    def seed(self, seed=None):  # legacy compatibility
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        return [int(seed) if seed is not None else int(getattr(Config, 'SEED', 0))]
