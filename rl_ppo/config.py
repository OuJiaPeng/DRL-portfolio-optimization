from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Project root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ETF_DATA_PATH: Path = BASE_DIR / "data" / "etf_data_with_indicators.csv"

    # ETF universe (column order must match data columns)
    ETF_TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'GLD', 'USO']

    # Date splits
    TRAIN_START = '2019-01-01'
    TRAIN_END   = '2024-05-31'
    VAL_START   = '2024-06-01'
    VAL_END     = '2024-12-31'
    TEST_START  = '2025-01-01'
    TEST_END    = '2025-07-01'

    # Risk-free rate
    RISK_FREE_ANNUAL: float = 0.04
    RISK_FREE_DAILY: float = 0.04 / 252


    # RL hyperparameters
    SEED = 42
    N_ENVS = 1
    N_STEPS = 4096
    BATCH_SIZE = 512
    N_EPOCHS = 5
    GAMMA = 0.995
    CLIP_RANGE = 0.2
    LEARNING_RATE = 2e-4
    DEVICE: str = "auto"  # "cuda", "cpu", or "auto"

    # Scheduling toggles
    USE_LR_SCHEDULE: bool = False
    USE_CLIP_SCHEDULE: bool = False

    # Policy architecture (pass to PPO via policy_kwargs)
    #e.g., {"pi": [256, 256], "vf": [256, 256]}
    POLICY_NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}
    ACTIVATION_FN = "relu"  

    # Training length
    TOTAL_TIMESTEPS = 300_000

    # Diagnostics
    METRICS_CSV_NAME: str = "training_progress.csv"
    EXPORT_DIAGNOSTICS: bool = True
    DIAGNOSTICS_DIR: Path = BASE_DIR / "results" / "diagnostics"

    # Trading frictions
    TURNOVER_COST: float = 0.00025  # refit overrides to 0.0001
    REBALANCE_FREQ: int = 1

    # Reward shaping
    REWARD_SCALE: float = 1.0
    MOMENTUM_ALPHA: float = 0
    MOMENTUM_LOOKBACK: int = 20
    REWARD_USE_EXCESS_RET: bool = True
    RISK_PENALTY_ALPHA: float = 0.001
    RISK_PENALTY_WINDOW: int = 20

    # Feature engineering defaults
    STACK_LEN: int = 10
    NORM_WINDOW: int = 63

    # PPO extras
    # Entropy anneal (start high for exploration, decay to low)
    ENT_COEF_INITIAL: float = 0.02
    ENT_COEF_FINAL: float = 0.002
    ENT_COEF_ANNEAL_STEPS: int = 80_000
    ENT_COEF: float = 0.02
    GAE_LAMBDA: float = 0.95
    MAX_GRAD_NORM: float = 0.5
    VF_COEF: float = 0.5
    ORTHO_INIT: bool = True

    # PPO stability/exploration knobs
    # KL anneal (prevent early choke, tighten later)
    TARGET_KL_INITIAL: float = 0.20
    TARGET_KL_FINAL: float = 0.05
    TARGET_KL_ANNEAL_STEPS: int = 100_000
    TARGET_KL_DELAY_STEPS: int = 40_000
    TARGET_KL: float | None = None
    USE_SDE: bool = True
    SDE_SAMPLE_FREQ: int = 4

    # Action smoothing & diversification
    ACTION_TEMPERATURE: float = 1.35

    # Legacy linear diversity shaping now disabled in favor of two-sided HHI band
    # DIVERSITY_BONUS_ALPHA: float = 0.0
    # DIVERSITY_BONUS_FINAL: float = 0.0
    # DIVERSITY_ANNEAL_STEPS: int = 0
    # DIVERSITY_MODE: str = "linear"
    # TARGET_HHI: float = 0.18
    # HHI_BAND: float = 0.06

    DIVERSITY_BONUS_ALPHA: float = 0.0
    INITIAL_LOGIT_CLIP: float = 4.0
    FINAL_LOGIT_CLIP: float = 3.0
    LOGIT_CLIP_ANNEAL_STEPS: int = 60_000
    LOGIT_CLIP: float = 4.0
    LOGIT_L2_PENALTY: float = 0.0
    # Debug action logging (disabled for final release)
    DEBUG_LOG_ACTIONS: bool = False
    DEBUG_LOG_LIMIT: int = 0
    DEBUG_LOG_PATH: Path = BASE_DIR / "results" / "action_debug.csv"  # retained for optional future re-enable

    # Diagnostic destabilizers to break uniform weight attractor
    INCLUDE_PREV_WEIGHTS: bool = True
    ZERO_PREV_WEIGHTS: bool = False
    RANDOM_INIT_WEIGHTS: bool = True
    DISABLE_TURNOVER_COST: bool = False
    MOVE_BONUS_COEF: float = 0.01
    DIAGNOSTIC_MODE: bool = False

    # Reward normalization (stabilize advantage scale across regimes)
    REWARD_NORMALIZE: bool = True
    REWARD_STD_WINDOW: int = 60
    REWARD_CENTER_MEAN: bool = False

    # Effective number of holdings (ENH) penalty to discourage ultra concentrated block allocations
    # ENH = 1 / HHI 
    MIN_ENH: float = 4.0
    ENH_PENALTY_COEF: float = 0.0

    # New two-sided diversification scheme (anti-uniform + anti-overconcentration)
    DIVERSITY_SCHEME: str = "two_sided_band"  # options: none, linear, two_sided_band
    HHI_LOWER_BAND: float = 0.13
    HHI_UPPER_BAND: float = 0.28
    UNI_TOO_LOW_COEF: float = 0.05
    CONC_PEN_COEF: float = 0.02
    BAND_CENTER_BONUS: float = 0.0
    ADV_COEF: float = 0.01

    # Asset-specific bias added to action logits each episode (symmetry breaker)
    USE_ASSET_BIAS: bool = True
    ASSET_BIAS_STD: float = 0.05

    # Position constraints  
    MAX_POSITION_SIZE: float = 0.35
    MIN_POSITION_SIZE: float = 0.0

    # Evaluation options
    EVAL_PREFER_BEST: bool = True  # load ./models/best_model.zip if present for final test eval
    # Validation evaluation settings
    VAL_N_WINDOWS: int = 3          # split validation period into this many chronological windows
    VAL_EVAL_FREQ: int = 5_000      # timesteps between eval passes
    VAL_EARLY_STOP_PATIENCE: int = 15  # eval passes with no mean Sharpe improvement before stop
    
    # Soft worst-window penalty (instead of hard veto)
    ENABLE_SOFT_GUARD: bool = True
    WORST_GUARD_THRESHOLD: float = -1.0
    SOFT_GUARD_LAMBDA: float = 0.25
    REQUIRE_WORST_ABOVE: bool = False
    MIN_WINDOW_SHARPE: float = -2.5

    # Feature engineering toggles (for extended features roadmap)
    FEAT_ADD_RET_HORIZONS: bool = True
    FEAT_RET_HORIZONS = (1, 5, 21, 63)
    FEAT_ADD_REALIZED_VOL: bool = True
    FEAT_VOL_WINDOWS = (5, 21, 63)
    FEAT_ADD_XS_RANKS: bool = True
    FEAT_ADD_MEAN_CORR: bool = True
    FEAT_CORR_WINDOW: int = 21
    FEAT_ADD_DOWNSIDE_SEMIVOL: bool = True
    FEAT_SEMIVOL_WINDOWS = (21, 63)
    FEAT_ADD_TIME_CYCLICAL: bool = True
    FEAT_ADD_ABS_RET: bool = True