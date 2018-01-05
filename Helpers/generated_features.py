import pandas as pd


def features_from_OHLC(spark_df, spark):
    """ Generate features for a stock/index based on 
           historical price and performance
    Args:
        df (dataframe with columns "Open", "Close", "High", 
               "Low", "Volume", "Adjusted Close")
    Returns:
        dataframe, data set with new features
    """
    df = spark_df
    df = df.toPandas()

    # 31 original features
    # average price
    df['avg_price_5'] = pd.rolling_mean(df['Close'], window=5).shift(1)
    df['avg_price_30'] = pd.rolling_mean(df['Close'], window=21).shift(1)
    df['avg_price_365'] = pd.rolling_mean(df['Close'], window=252).shift(1)
    df['ratio_avg_price_5_30'] = df['avg_price_5'] / df['avg_price_30']
    df['ratio_avg_price_5_365'] = df['avg_price_5'] / df['avg_price_365']
    df['ratio_avg_price_30_365'] = df['avg_price_30'] / df['avg_price_365']
    # average volume
    df['avg_volume_5'] = pd.rolling_mean(df['Volume'], window=5).shift(1)
    df['avg_volume_30'] = pd.rolling_mean(df['Volume'], window=21).shift(1)
    df['avg_volume_365'] = pd.rolling_mean(df['Volume'], window=252).shift(1)
    df['ratio_avg_volume_5_30'] = df['avg_volume_5'] / df['avg_volume_30']
    df['ratio_avg_volume_5_365'] = df['avg_volume_5'] / df['avg_volume_365']
    df['ratio_avg_volume_30_365'] = df['avg_volume_30'] / df['avg_volume_365']
    # standard deviation of prices
    df['std_price_5'] = pd.rolling_std(df['Close'], window=5).shift(1)
    df['std_price_30'] = pd.rolling_std(df['Close'], window=21).shift(1)
    df['std_price_365'] = pd.rolling_std(df['Close'], window=252).shift(1)
    df['ratio_std_price_5_30'] = df['std_price_5'] / df['std_price_30']
    df['ratio_std_price_5_365'] = df['std_price_5'] / df['std_price_365']
    df['ratio_std_price_30_365'] = df['std_price_30'] / df['std_price_365']
    # standard deviation of volumes
    df['std_volume_5'] = pd.rolling_std(df['Volume'], window=5).shift(1)
    df['std_volume_30'] = pd.rolling_std(df['Volume'], window=21).shift(1)
    df['std_volume_365'] = pd.rolling_std(df['Volume'], window=252).shift(1)
    df['ratio_std_volume_5_30'] = df['std_volume_5'] / df['std_volume_30']
    df['ratio_std_volume_5_365'] = df['std_volume_5'] / df['std_volume_365']
    df['ratio_std_volume_30_365'] = df['std_volume_30'] / df['std_volume_365']
    # return
    df['return_1'] = ((
                          df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df['return_5'] = ((
                          df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df['return_30'] = ((
                           df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df['moving_avg_5'] = pd.rolling_mean(df['return_1'], window=5)
    df['moving_avg_30'] = pd.rolling_mean(df['return_1'], window=21)
    df['moving_avg_365'] = pd.rolling_mean(df['return_1'], window=252)
    df = df.dropna(axis=0)
    result_df = spark.createDataFrame(df.round(3))
    return result_df
