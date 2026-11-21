## Creating target and features

## Features : for every serie of T days, our model will take a non-profit setting, a stop-loss, and a time limit, if the stock price reaches the non profit, it outputs 1, if it reaches stop loss, it outputs -1, if it reaches time limit without doing both, then it outputs 0.

## Lewts first create the functions that will determine Ytrue, the true targets that our machine modelling will train on, and be evaluated on.

## Lets split our datas in subsets including the t days given, to predict wether the value will reach the stop loss SL, or the non profit NP in T days maximum.
# --> We need to split our dataset in series of T* = T + t days.

import numpy as np
import pandas as pd

from .metadata import Co_datas, Variation

from sklearn.preprocessing import StandardScaler

def feature_target(df, T, t, SL, NP, SMA=20, ATRS=20, ATRL=3, ZVOL=20):
    dco = Co_datas(df, SMA, ATRS, ATRL, ZVOL)
    dvar = Variation(df)
    DF = pd.concat([dco, dvar], axis=1)
    R = len(df)
    prog = 1

    Xi_train_list = []
    Xi_test_list = []
    Yi_train_list = []
    Yi_test_list = []

    train_limit = int((R - t - T) * 0.8)
    embargo_gap = t + T

    for I in range(0, max(0, R - t - T + 1), 5):
        prc = I / R * 100
        if prc > prog * 10:  ## print progression
            print(prc, ' / 100')
            prog = prog + 1

        if I + t > len(dco):
            break

        if I < train_limit:  ## Form train set 80%
            Xtrain = DF.iloc[I:I+t].copy()  ## The serie of t values at step i
            close_train = df.loc[Xtrain.index, 'close'].values

            Xtrain["stoploss"] = close_train - Xtrain["atr"].values * SL
            Xtrain["nonprofit"] = close_train + Xtrain["atr"].values * NP
            Xtrain["timelimit"] = T

            start_df_idx = df.index.get_loc(Xtrain.index[0])
            Xtrain["_df_idx"] = np.arange(start_df_idx, start_df_idx + len(Xtrain))

            ## Creating Ytrain
            Ytrain = pd.Series(1, index=Xtrain.index)
            for j in range(len(Xtrain)):  # Creating Ytrain from 0 to its t -th value
                RL = Xtrain.index[j]
                current_df_idx = start_df_idx + j
                for k in range(current_df_idx + 1, min(len(df), current_df_idx + T + 1)):  # Checking if limits reached
                    if k >= len(df):
                        break
                    if df['close'].iloc[k] < Xtrain.loc[RL, 'stoploss']:
                        Ytrain.loc[RL] = 0
                        break
                    elif df['close'].iloc[k] > Xtrain.loc[RL, 'nonprofit']:
                        Ytrain.loc[RL] = 2
                        break

            Xi_train_list.append(Xtrain)
            Yi_train_list.append(Ytrain.values)

        elif I >= train_limit + embargo_gap:
            Xtest = DF.iloc[I:I+t].copy()  ## The serie of t values at step i
            close_test = df.loc[Xtest.index, 'close'].values

            Xtest["stoploss"] = close_test - Xtest["atr"].values * SL
            Xtest["nonprofit"] = close_test + Xtest["atr"].values * NP
            Xtest["timelimit"] = T

            start_df_idx = df.index.get_loc(Xtest.index[0])
            Xtest["_df_idx"] = np.arange(start_df_idx, start_df_idx + len(Xtest))

            ## Creating Ytest
            Ytest = pd.Series(1, index=Xtest.index)
            for j in range(len(Xtest)):  # Creating Ytest from 0 to its t -th value
                RL = Xtest.index[j]
                current_df_idx = start_df_idx + j
                for k in range(current_df_idx + 1, min(len(df), current_df_idx + T + 1)):  # Checking if limits reached
                    if k >= len(df):
                        break
                    if df['close'].iloc[k] < Xtest.loc[RL, 'stoploss']:
                        Ytest.loc[RL] = 0
                        break
                    elif df['close'].iloc[k] > Xtest.loc[RL, 'nonprofit']:
                        Ytest.loc[RL] = 2
                        break

            Xi_test_list.append(Xtest)
            Yi_test_list.append(Ytest.values)

    Xi_train = pd.concat(Xi_train_list, axis=0, ignore_index=True) if Xi_train_list else pd.DataFrame()
    Xi_test = pd.concat(Xi_test_list, axis=0, ignore_index=True) if Xi_test_list else pd.DataFrame()
    Yi_train = np.concatenate(Yi_train_list) if Yi_train_list else np.array([])
    Yi_test = np.concatenate(Yi_test_list) if Yi_test_list else np.array([])

    return (Xi_train,
            Yi_train,
            Xi_test,
            Yi_test)

def standardize(df, T, t, SL, NP, SMA=20, ATRS=20, ATRL=3, ZVOL=20):
    (Xi_train, Yi_train, Xi_test, Yi_test) = feature_target(df, T, t, SL, NP, SMA, ATRS, ATRL, ZVOL)

    scaler = StandardScaler()
    # Drop metadata columns before scaling features
    meta_cols = ['_df_idx', 'stoploss', 'nonprofit', 'timelimit']
    train_features = Xi_train.drop(columns=[c for c in meta_cols if c in Xi_train.columns], errors='ignore')
    test_features = Xi_test.drop(columns=[c for c in meta_cols if c in Xi_test.columns], errors='ignore')

    Xi_train_scaled = scaler.fit_transform(train_features)
    Xi_test_scaled = scaler.transform(test_features)

    # Return Xi_test as well so we retain the metadata and indices!
    return (Xi_train_scaled, Yi_train, Xi_test_scaled, Yi_test, Xi_test)