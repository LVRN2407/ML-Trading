import numpy as np
import pandas as pd
from math import sqrt

from ..data_treatment.metadata import Numerize


def sharpe_ratio(Xtest, df, Ypred, Ytest, T, Ts, fee_pct=0.001,
                 trading_days_per_year=252, skip_overlapping_trades=True):
    trade = 0
    df = Numerize(df)
    Predval = Ypred.tolist()
    Testval = Ytest.tolist()
    Returns = []
    Trade_Returns = []

    df_close_arr = df['close'].values

    # Handle Xtest safely whether it's a DataFrame or a NumPy array
    if hasattr(Xtest, 'columns'):
        x_stop_arr = Xtest['stoploss'].values if 'stoploss' in Xtest.columns else np.zeros(len(Xtest))
        x_np_arr = Xtest['nonprofit'].values if 'nonprofit' in Xtest.columns else np.zeros(len(Xtest))
        df_indices = Xtest['_df_idx'].values if '_df_idx' in Xtest.columns else np.arange(len(Xtest))
    else:
        # Fallback if Xtest was converted to a raw numpy array
        Xtest = np.array(Xtest)
        x_stop_arr = np.zeros(len(Xtest))
        x_np_arr = np.zeros(len(Xtest))
        df_indices = np.arange(len(Xtest))

    skip_counter = 0

    for i in range(len(Predval)):
        # If a trade is currently active, skip taking new ones to prevent overlapping inflation
        if skip_overlapping_trades and skip_counter > 0:
            skip_counter -= 1
            Returns.append(0.0)
            continue

        pred = Predval[i]
        df_idx = int(df_indices[i])

        if df_idx >= len(df_close_arr):
            Returns.append(0.0)
            continue

        P0 = float(df_close_arr[df_idx])
        profit = 0.0

        if pred == 2:  # Model chooses to invest
            trade = trade + 1
            if Testval[i] == 0:  # Stoploss hit
                profit = (float(x_stop_arr[i]) - P0) / P0
            elif Testval[i] == 1:  # Time limit hit
                future_idx = df_idx + T
                if future_idx < len(df_close_arr):
                    profit = (float(df_close_arr[future_idx]) - P0) / P0
                else:
                    profit = 0.0
            elif Testval[i] == 2:  # Nonprofit hit
                profit = (float(x_np_arr[i]) - P0) / P0

            # Apply friction to the executed trade
            profit = profit - fee_pct

            # Lock out subsequent entries for T bars to enforce independence
            skip_counter = T

            Trade_Returns.append(float(profit))

        Returns.append(float(profit))

    Returns = pd.Series(Returns, dtype=float)
    Trade_Returns = pd.Series(Trade_Returns, dtype=float)

    R = Trade_Returns.rolling(Ts, min_periods=1).mean()
    r = Trade_Returns.mean()

    S = Trade_Returns.rolling(Ts, min_periods=1).std()
    s = Trade_Returns.std()

    Tradesperyear = trade/len(Predval) * trading_days_per_year

    SHR = []
    for j in range(len(R)):
        std_val = S.iloc[j] if hasattr(S, 'iloc') else S[j]
        mean_val = R.iloc[j] if hasattr(R, 'iloc') else R[j]
        if std_val == 0 or pd.isna(std_val):
            SR = 0.0
        else:
            SR = (mean_val / std_val) * sqrt(Tradesperyear)  # Annualizing
        SHR.append(SR)



    SHR = pd.Series(SHR)
    shr = (r / s) * sqrt(Tradesperyear) if trade >= 2 and s != 0 and not pd.isna(s) else 0.0

    return SHR, shr
