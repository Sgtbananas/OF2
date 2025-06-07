import numpy as np

def add_all_features(df):
    """
    Add all features required by the trained ML model to the DataFrame.
    Must match training code exactly!
    """
    # ATR 14
    if 'high' in df and 'low' in df and 'close' in df:
        df['tr'] = np.maximum(df['high'] - df['low'],
                              np.abs(df['high'] - df['close'].shift(1)),
                              np.abs(df['low'] - df['close'].shift(1)))
        df['atr14'] = df['tr'].rolling(window=14).mean()
    # Realized volatility and other statistics
    if 'close' in df:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_10'] = df['log_return'].rolling(window=10).std()
        df['return_3'] = df['close'].pct_change(3)
    for w in [5, 10, 20]:
        if 'close' in df:
            df[f'roll_close_std_{w}'] = df['close'].rolling(window=w).std()
        if 'volume' in df:
            df[f'roll_vol_mean_{w}'] = df['volume'].rolling(window=w).mean()
            df[f'roll_vol_std_{w}']  = df['volume'].rolling(window=w).std()
    # Add dummy values for features like entry_idx, exit_idx, pnl (set to 0.0 for live)
    for col in ['entry_idx', 'exit_idx', 'pnl']:
        if col not in df:
            df[col] = 0.0
    df.fillna(0, inplace=True)
    return df