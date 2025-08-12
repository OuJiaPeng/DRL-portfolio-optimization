import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys



def load_full_feature_data(csv_path: str | None = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.environ.get("ETF_FEATURES_CSV")
    candidates = []
    if csv_path:
        candidates.append(csv_path)
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(base_dir, 'etf_data_with_indicators.csv'))
    candidates.append(os.path.join(os.path.dirname(base_dir), 'etf_data_with_indicators.csv'))
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'data', 'etf_data_with_indicators.csv'))
    data_path = None
    for p in candidates:
        if p and os.path.exists(p):
            data_path = p
            break
    if not data_path:
        return None
    data_df = pd.read_csv(data_path, index_col=0, header=[0, 1], parse_dates=True)
    return data_df

def _parse_date_window(arg: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None] | None:
    if not arg:
        return None
    try:
        start_s, end_s = (arg.split(":", 1) if ":" in arg else ("", arg))
        start = pd.to_datetime(start_s) if start_s else None
        end = pd.to_datetime(end_s) if end_s else None
        return (start, end)
    except Exception:
        return None

def _apply_date_window(df: pd.DataFrame, window: tuple[str | None, str | None] | tuple[pd.Timestamp | None, pd.Timestamp | None] | None) -> pd.DataFrame:
    if window is None:
        return df
    start, end = window
    if isinstance(start, str) and start:
        start = pd.to_datetime(start)
    if isinstance(end, str) and end:
        end = pd.to_datetime(end)
    try:
        if start is not None and end is not None:
            return df.loc[start:end]
        if start is not None:
            return df.loc[start:]
        if end is not None:
            return df.loc[:end]
        return df
    except Exception:
        return df

def extract_features_for_correlation(data_df, etf='SPY'):
    if data_df is None:
        return None
    etf_columns = [col for col in data_df.columns if col[0] == etf]
    etf_data = data_df[etf_columns].copy()
    etf_data.columns = [col[1] for col in etf_data.columns]
    if 'returns' not in etf_data.columns and 'close' in etf_data.columns:
        etf_data['returns'] = etf_data['close'].pct_change()
    indicators = [
        'close', 'volume', 'returns', 'open', 'high', 'low',
        'rsi_14', 'macd', 'signal', 'macd_diff', 'ema_20'
    ]
    available_indicators = [ind for ind in indicators if ind in etf_data.columns]
    feature_data = etf_data[available_indicators].copy()
    feature_data = feature_data.ffill().fillna(0)
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feature_data

def create_correlation_heatmap(feature_data, etf='SPY', save_dir=None):
    if feature_data is None or feature_data.empty:
        return None, []
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    corr_matrix = feature_data.corr()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(im)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title(f'Technical Indicator Correlation Matrix - {etf}\n(Used for RL Feature Selection)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Technical Indicators', fontsize=12)
    plt.ylabel('Technical Indicators', fontsize=12)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'feature_correlation_analysis_{etf.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Select initial feature for minimal set
    selected_features = []
    if 'returns' in feature_data.columns:
        selected_features.append('returns')
    if not selected_features and 'close' in feature_data.columns:
        selected_features.append('close')
    if not selected_features:
        selected_features.append(feature_data.columns[0])
    remaining_features = [f for f in feature_data.columns if f not in selected_features]
    for feat in remaining_features:
        max_corr = max([abs(corr_matrix.loc[feat, sel_feat]) for sel_feat in selected_features])
        if max_corr < 0.6:
            selected_features.append(feat)
            if len(selected_features) >= 5:
                break
    plt.close()
    return corr_matrix, selected_features

def analyze_all_etfs(save_dir=None, csv_path: str | None = None, date_window: tuple[str | None, str | None] | None = None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    data_df = load_full_feature_data(csv_path=csv_path)
    if data_df is None:
        return
    if date_window is not None:
        data_df = _apply_date_window(data_df, date_window)
        if data_df.empty:
            return
    etfs = list(set([col[0] for col in data_df.columns]))
    all_recommendations = {}
    for etf in etfs:
        feature_data = extract_features_for_correlation(data_df, etf)
        _, recommendations = create_correlation_heatmap(feature_data, etf, save_dir)
        all_recommendations[etf] = recommendations
    if all_recommendations:
        print("\n" + "="*60)
        print("SUMMARY ACROSS ALL ETFS")
        print("="*60)
        all_features = []
        for etf_recs in all_recommendations.values():
            all_features.extend(etf_recs)
        feature_counts = pd.Series(all_features).value_counts()
        print("Most commonly recommended features:")
        for feat, count in feature_counts.head(10).items():
            print(f"   {feat}: recommended by {count}/{len(all_recommendations)} ETFs")
    return all_recommendations

if __name__ == "__main__":
    DEFAULT_DATE_WINDOW = "2016-01-01:2024-12-31"
    user_csv = sys.argv[1] if len(sys.argv) > 1 else None
    user_window_arg = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("ANALYSIS_DATE_WINDOW")
    cli_date_window = _parse_date_window(user_window_arg) if user_window_arg else None
    effective_window = cli_date_window
    out_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_all_etfs(save_dir=out_dir, csv_path=user_csv, date_window=effective_window)
