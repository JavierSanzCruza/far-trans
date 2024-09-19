import math

import numpy as np
from beta_rec.utils.constants import DEFAULT_RATING_COL


def avg_price(df, periods=(21, 63, 126, 189)):
    """
    Computes the average price of an asset over a period.
    :param df: a dataframe containing the price of an asset over a given period.
    :param periods: the periods to consider.
    :return: the average price of the assets for the different periods.
    """
    for t in periods:
        df[f"avg_price_{t}d"] = df[DEFAULT_RATING_COL].rolling(window=t).mean()
    return df


def roi(df, periods=(1, 21, 63, 126, 189)):
    """
    Computes the return on investment of an asset over some given periods:
    :param df: a dataframe containing the prices of an asset over different periods.
    :param periods:
    :return:
    """
    for t in periods:
        df[f"past_profitability_{t}d"] = (df[DEFAULT_RATING_COL] - df[DEFAULT_RATING_COL].shift(t)) / df[DEFAULT_RATING_COL].shift(t)
    return df


def volatility(df, periods=(21, 63, 126, 189)):
    """
    Computes the volatility of an asset over a given period.
    :param df: a dataframe containing the prices of an asset.
    :param periods:
    :return:
    """
    drop_col = False
    if "past_profitability_1d" not in df:
        df["past_profitability_1d"] = (df[DEFAULT_RATING_COL] - df[DEFAULT_RATING_COL].shift(1)) / df[DEFAULT_RATING_COL].shift(1)
        drop_col = True

    for t in periods:
        if t == 0:
            df[f"volatility"] = df["past_profitability_1d"].expanding(min_periods=2).std() * np.sqrt(252)
        else:
            df[f"volatility_{t}d"] = df["past_profitability_1d"].rolling(window=t).std() * np.sqrt(252)
    if drop_col:
        df = df.drop(columns=["past_profitability_1d"])
    return df


def sharpe(df, periods=(21, 63, 126, 189)):
    for t in periods:
        drop_vol = False
        drop_roi = False
        if f"volatility_{t}d" not in df:
            volatility(df, [t])
            drop_vol = True
        if f"past_profitability_{t}d" not in df:
            roi(df, [t])
            drop_roi = True
        df[f"sharpe_{t}d"] = df[f"past_profitability_{t}d"]/df[f"volatility_{t}d"]
        if drop_vol:
            df = df.drop(columns=[f"volatility_{t}d"])
        if drop_roi:
            df = df.drop(columns=[f"past_profitability_{t}d"])
        df[f"sharpe_{t}d"] = df[f"sharpe_{t}d"].apply(lambda x: 0.0 if math.isnan(x) else x)
        df[f"sharpe_{t}d"] = df[f"sharpe_{t}d"].apply(lambda x: 0.0 if math.isinf(x) else x)
    return df


def moving_average_convergence_divergence(df):
    close_EMA_26 = df[DEFAULT_RATING_COL].ewm(span=26, adjust=False).mean()
    close_EMA_12 = df[DEFAULT_RATING_COL].ewm(span=12, adjust=False).mean()

    df['MACD'] = close_EMA_12 - close_EMA_26
    return df


def momentum(df, periods=(21, 63, 126, 189)):
    for t in periods:
        df[f"m_{t}d"] = df[DEFAULT_RATING_COL].diff(t)
    return df


def rate_of_change(df, periods=(21, 63, 126, 189)):

    for t in periods:
        if f"m_{t}d" not in df:
            momentum(df, periods)
            break

    for t in periods:
        df[f"roc_{t}d"] = df[f"m_{t}d"] / df[DEFAULT_RATING_COL].shift(t)
    return df


def relative_strength_index(df, n=14):
    u = df[DEFAULT_RATING_COL].diff()
    d = df[DEFAULT_RATING_COL].shift(1) - df[DEFAULT_RATING_COL]
    df['up'] = np.where(u > 0, u, 0)
    df['down'] = np.where(d > 0, d, 0)
    rsi_name = 'rsi_' + str(n)
    df[rsi_name] = 100 - 100 / (
                1 + df['up'].ewm(span=n, adjust=False).mean() / df['down'].ewm(span=n, adjust=False).mean())
    df[rsi_name] = df[rsi_name].apply(lambda x: 0.0 if math.isnan(x) else x)
    df = df.drop(['up', 'down'], axis=1)
    return df


def detrended_close_oscillator(df, n=22):
    dco_name = 'dco_' + str(n)
    mid_index = int(n / 2 + 1)
    df[dco_name] = df[DEFAULT_RATING_COL].shift(mid_index) - df[DEFAULT_RATING_COL].rolling(window=n).mean()
    return df

def min_max(df, periods=(21, 63, 126, 189)):
    for t in periods:
        df[f"min_{t}d"] = df[DEFAULT_RATING_COL].rolling(window=t).min()
        df[f"max_{t}d"] = df[DEFAULT_RATING_COL].rolling(window=t).max()

        df[f'exp_mean_{t}d'] = df[DEFAULT_RATING_COL].ewm(span=t).mean()

    return df

