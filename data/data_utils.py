import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from rl_ppo.config import Config


def load_and_process_data() -> pd.DataFrame:
    """
    Load the multi-index ETF CSV and extract close prices per ticker.
    Returns a DataFrame indexed by date with columns ordered as Config.ETF_TICKERS.
    """
    df = pd.read_csv(Config.ETF_DATA_PATH, header=[0, 1], index_col=0, parse_dates=True).sort_index()
    closes = {sym: df[(sym, 'close')] for sym in Config.ETF_TICKERS if (sym, 'close') in df.columns}
    missing = [sym for sym in Config.ETF_TICKERS if (sym, 'close') not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) for close price: {missing}")
    prices = pd.DataFrame(closes, index=df.index).sort_index().ffill().dropna()
    return prices


def create_features(
    prices: pd.DataFrame,
    stack_len: Optional[int] = None,
    norm_window: Optional[int] = None,
    fit_end_date: Optional[Union[str, pd.Timestamp]] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Defaults from Config if not provided
    stack_len = stack_len if stack_len is not None else getattr(Config, "STACK_LEN", 10)
    norm_window = norm_window if norm_window is not None else getattr(Config, "NORM_WINDOW", 63)

    # 1) Log-returns and rolling z-score (baseline block)
    log_ret = np.log(prices / prices.shift(1))
    mu_full = log_ret.rolling(norm_window, min_periods=norm_window).mean()
    sigma_full = log_ret.rolling(norm_window, min_periods=norm_window).std(ddof=0)
    if fit_end_date is not None:
        cutoff = pd.to_datetime(fit_end_date)
        mu_hist = mu_full.loc[:cutoff]
        sigma_hist = sigma_full.loc[:cutoff]
        mu = mu_hist.reindex(mu_full.index).ffill()
        sigma = sigma_hist.reindex(sigma_full.index).ffill()
    else:
        mu = mu_full
        sigma = sigma_full
    r_norm = (log_ret - mu) / (sigma + 1e-8)

    frames = []
    col_names: List[str] = []
    # A) Frame-stacked z-scored log-returns (lags 0..stack_len-1)
    for lag in range(stack_len):
        shifted = r_norm.shift(lag)
        frames.append(shifted)
        for c in prices.columns:
            col_names.append(f"zlogret_{c}_t-{lag}")

    # 2) Momentum features: 20D and 60D simple return
    mom20 = prices / prices.shift(20) - 1.0
    mom60 = prices / prices.shift(60) - 1.0
    frames += [mom20, mom60]
    for c in prices.columns:
        col_names.append(f"mom20_{c}")
    for c in prices.columns:
        col_names.append(f"mom60_{c}")

    # 3) RSI(14) computed from closes (Wilder's smoothing approximation via EMA)
    diff = prices.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    # Use EMA with alpha=1/14 as a proxy for Wilder's average
    span = 14
    alpha = 1.0 / span
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=span).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=span).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))
    frames.append(rsi14)
    for c in prices.columns:
        col_names.append(f"rsi14_{c}")

    # 4) Volume features (optional, from provided raw_df)
    vol_df = None
    if raw_df is not None:
        vols = {sym: raw_df[(sym, 'volume')] for sym in Config.ETF_TICKERS if (sym, 'volume') in raw_df.columns}
        if vols:
            vol_df = pd.DataFrame(vols, index=raw_df.index).sort_index().loc[prices.index]

    if vol_df is not None and not vol_df.empty:
        vol_log = np.log1p(vol_df)
        vol_mu = vol_log.rolling(norm_window, min_periods=norm_window).mean()
        vol_sigma = vol_log.rolling(norm_window, min_periods=norm_window).std(ddof=0)
        vol_z = (vol_log - vol_mu) / (vol_sigma + 1e-8)
        frames.append(vol_z)
        for c in prices.columns:
            col_names.append(f"volz_{c}")

    # Combine all blocks
    feat = pd.concat(frames, axis=1)
    feat.columns = col_names

    # Align and drop NaNs arising from lookbacks
    feat = feat.dropna().copy()
    prices_aligned = prices.loc[feat.index].copy()
    return feat, prices_aligned


def train_val_test_split(prices: pd.DataFrame):
    idx = prices.index
    ts = pd.to_datetime(Config.TRAIN_START)
    te = pd.to_datetime(getattr(Config, 'TRAIN_END', Config.VAL_START))
    vs = pd.to_datetime(Config.VAL_START)
    ve = pd.to_datetime(getattr(Config, 'VAL_END', Config.TEST_START))
    tes = pd.to_datetime(Config.TEST_START)
    tee = pd.to_datetime(Config.TEST_END)
    train_mask = (idx >= ts) & (idx <= te)
    val_mask = (idx >= vs) & (idx <= ve)
    test_mask = (idx >= tes) & (idx <= tee)
    return prices[train_mask], prices[val_mask], prices[test_mask]


__all__ = ["load_and_process_data", "create_features", "train_val_test_split"]
