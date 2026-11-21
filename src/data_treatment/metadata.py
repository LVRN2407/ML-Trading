import numpy as np
import pandas as pd


def Numerize(df):
    return df.reset_index(drop=True)


def Variation(dfnum):
    dfvar = dfnum.copy()
    for col in dfnum.columns:
        dfvar[col] = dfnum[col].pct_change()
    dfvar.columns = dfnum.columns + ' Variation '
    return dfvar[1:]

def Extend_to_delay(df, d):
    df_extended = df.copy()
    df_shifted = df.shift(-d)

    # Rename the shifted columns to clearly distinguish them
    df_shifted.columns = [str(col) + f'_lead_{d}' for col in df_shifted.columns]

    # Combine them side-by-side (indices match 1:1 automatically)
    return pd.concat([df_extended, df_shifted], axis=1).dropna()


## Adding new variables

def Co_datas(df, SMA=20, ATRS=3, ATRL=20, ZVOL=20):
    H = df['high']
    L = df['low']
    C = df['close']
    O = df['open']
    V = df['volume']
    dvar = df.copy()

    dvar['log_return'] = np.log(C/C.shift(1).replace(0, np.nan))

    sma = df.close.rolling(window=SMA).mean()
    dvar['dist_sma'] = (C - sma)/sma.replace(0, np.nan)

    dvar['garman_klass'] = np.sqrt(0.5*np.log(H/L.replace(0, np.nan))**2)

    high_low = H - L
    high_close = abs(H - C.shift(1))
    low_close = abs(L - C.shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atrs = tr.rolling(window=ATRS).mean()
    atrl = tr.rolling(window=ATRL).mean()
    dvar['atr'] = atrs
    dvar['atr_ratio'] = atrs / atrl.replace(0, np.nan)

    stdvol = V.rolling(window=ZVOL).std()
    volmean = V.rolling(window=ZVOL).mean()
    dvar['vol_zscore'] = (V - volmean)/stdvol.replace(0, np.nan)

    dvar['vwap_distance'] = (C - df['vwap'])/df['vwap'].replace(0, np.nan)
    HL = H - L
    dvar['wick_ratio'] = (H - np.maximum(O,C))/HL.replace(0, np.nan)

    dvar['body_ratio'] = abs(C - O)/HL.replace(0, np.nan)

    return pd.DataFrame(dvar).drop(columns=df.columns).dropna()


