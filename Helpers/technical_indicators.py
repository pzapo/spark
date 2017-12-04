import pandas as pd


def RSI(dataframe, window_length, avg_type, spark, column='Close'):
    data = dataframe.toPandas()
    # Get just the close
    close = data['Close']
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    if avg_type == "EWMA":
        roll_up = up.ewm(span=window_length, min_periods=window_length).mean()
        roll_down = down.abs().ewm(
            span=window_length, min_periods=window_length).mean()
    elif avg_type == "SMA":
        roll_up = pd.rolling_mean(up, window_length)
        roll_down = pd.rolling_mean(down.abs(), window_length)
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    RSI = pd.DataFrame({'RSI': RSI})
    data = data.join(RSI)
    result_df = spark.createDataFrame(data.round(3))
    return result_df


# Commodity Channel Index
def CCI(spark_df, ndays, spark):
    data = spark_df.toPandas()
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    CCI = pd.Series(
        (TP - pd.rolling_mean(TP, ndays)) /
        (0.015 * pd.rolling_std(TP, ndays)),
        name='CCI')
    data = data.join(CCI)
    result_df = spark.createDataFrame(data.round(3))
    return result_df


# Moving average convergence divergence
def MACD(dataframe, spark, nfast=12, nslow=26, signal=9, column='Close'):
    data = dataframe.toPandas()
    # Get just the close
    price = data[column]
    # Get the difference in price from previous step
    emaslow = pd.ewma(price, span=nslow, min_periods=1)
    emafast = pd.ewma(price, span=nfast, min_periods=1)
    #     MACD = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    MACD = pd.DataFrame({'MACD': emafast - emaslow})

    data = data.join(MACD.round(3))
    result_df = spark.createDataFrame(data)
    return result_df
