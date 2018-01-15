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

    # average price
    df['avgPrice5'] = pd.rolling_mean(df['Close'], window=5).shift(1)
    df['avgPrice30'] = pd.rolling_mean(df['Close'], window=21).shift(1)
    df['ratioAvgPrice5_30'] = df['avgPrice5'] / df['avgPrice30']
    # # standard deviation of prices
    df['stdPrice5'] = pd.rolling_std(df['Close'], window=5).shift(1)
    df['stdPrice30'] = pd.rolling_std(df['Close'], window=21).shift(1)
    df['ratioStdPrice5_30'] = df['stdPrice5'] / df['stdPrice30']
    # average volume
    df['avgVolume5'] = pd.rolling_mean(df['Volume'], window=5).shift(1)
    df['avgVolume30'] = pd.rolling_mean(df['Volume'], window=21).shift(1)
    df['ratioAvgVolume5_30'] = df['avgVolume5'] / df['avgVolume30']
    # standard deviation of volumes
    df['stdVolume5'] = pd.rolling_std(df['Volume'], window=5).shift(1)
    df['stdVolume30'] = pd.rolling_std(df['Volume'], window=21).shift(1)
    df['ratioStdVolume5_30'] = df['stdVolume5'] / df['stdVolume30']
    # return
    df['return1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df['return5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df['return30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    # mvg avg
    df['movingAvg5'] = pd.rolling_mean(df['return1'], window=5)
    df['movingAvg30'] = pd.rolling_mean(df['return1'], window=21)
    df = df.dropna(axis=0)
    result_df = spark.createDataFrame(df.round(3))
    # return result_df

    return result_df.drop('stdPrice5',
'stdPrice30',
'OBV',
'ratioStdPrice5_30',
'ratioAvgPrice5_30',
'return30',
'avgVolume5',
'ratioAvgVolume5_30',
'MACD',
'stdVolume5',
'avgPrice30',
'movingAvg30',
'stdVolume30',
'ratioStdVolume5_30',
'movingAvg5',
'avgVolume30',
'avgPrice5')
