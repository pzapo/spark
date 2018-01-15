import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id, udf, split
from pyspark.sql.types import IntegerType, DoubleType

from Helpers.generated_features import features_from_OHLC
from Helpers.technical_indicators import calc_ti
from Helpers.udf import profit_, ReverseTradeClassifier, BuyAndHoldClassifier
from pyspark.sql.functions import col

profit_udf = udf(profit_, IntegerType())
BHC = udf(BuyAndHoldClassifier, DoubleType())
RC = udf(ReverseTradeClassifier, DoubleType())


def initial_processing(spark, path_to_csv):
    '''
    Initial dataframep processing,
    Removing null values, and converting to proper
    types
    :param spark:
    :param path_to_csv:
    :return:
    '''
    fresh_df = spark.read.csv(path_to_csv, header=True, inferSchema=True)
    # if ("Orlen" or "OrlenVerify") in path_to_csv:
    # fresh_df = fresh_df.filter(fresh_df.Open != "null")
    processed_df = fresh_df.select(fresh_df["Open"].cast("float"),
                                   fresh_df["High"].cast("float"), fresh_df["Volume"].cast("int"),
                                   fresh_df["Low"].cast("float"), fresh_df["Close"].cast("float"))
    processed_df = processed_df.select("*").withColumn(
        "id", monotonically_increasing_id())
    return processed_df


def calc_profit(df):
    '''
    Creating new column with shifted Close price by 1 day
    Profit label calculation
    1 if stock risen up, 0 is it went down
    :param df:
    :return:
    '''
    df_daily_return = df.withColumn('prev_day_price',
                                    F.lag(df['Open']).over(
                                        Window.orderBy("id")))

    df_daily_return = df_daily_return.filter(
        df_daily_return.prev_day_price.isNotNull())

    df_profit = df_daily_return.withColumn(
        'Profit', profit_udf(df_daily_return.Open,
                             df_daily_return.prev_day_price))

    df_shifted_profit = df_profit.withColumn(
        'Profit',
        F.lag(df_profit['Profit'], count=-1).over(Window.orderBy("id")))

    final_df = df_shifted_profit.filter(df_shifted_profit.Profit.isNotNull())

    final_df = final_df.drop("Daily return")
    final_df = final_df.drop("prev_day_price")

    return final_df


def calc_profit2(df):
    '''
    Creating new column with shifted Close price by 1 day
    Profit label calculation
    1 if stock risen up, 0 is it went down
    :param df:
    :return:
    '''
    df_daily_return = df.withColumn('prev_day_price',
                                    F.lag(df['Open']).over(
                                        Window.orderBy("id")))

    df_daily_return = df_daily_return.filter(
        df_daily_return.prev_day_price.isNotNull())

    df_profit = df_daily_return.withColumn(
        'Profit', profit_udf(df_daily_return.Open,
                             df_daily_return.prev_day_price))

    # df_profit = df_profit.withColumn(
    #     'prediction', BHC(df_profit.Profit, F.lag(df_profit['Profit']).over(
    #         Window.orderBy("id"))))

    df_profit = df_profit.withColumn(
        'prediction', RC(df_profit.Profit))

    df_shifted_profit = df_profit.withColumn(
        'Profit',
        F.lag(df_profit['Profit'], count=-1).over(Window.orderBy("id")))

    final_df = df_shifted_profit.filter(df_shifted_profit.Profit.isNotNull())

    final_df = final_df.drop("Daily return")
    final_df = final_df.drop("prev_day_price")
    return final_df


def transform_date(df):
    '''
    Convert date to splitted format
    year | month | day

    Columns without Date
    converted_df = converted_df.select(
        [col(c).cast('float') for c in converted_df.columns if c. not in {'Date'}])
    :param df:
    :return:
    '''
    df_date = df.select(df.Date)
    df_date = df_date.select("*").withColumn("id", monotonically_increasing_id())
    split_col = split(df['Date'], '-')
    df = df.withColumn('Year',
                       split_col.getItem(0).cast('int'))
    df = df.withColumn('Month',
                       split_col.getItem(1).cast('int'))
    df = df.withColumn('Day',
                       split_col.getItem(2).cast('int'))
    return df.drop("Date")


def train_test_split(spark, df, train_fold, test_fold, manual_split, random_seed):
    '''
    :param spark:
    :param df:
    :param CHUNKS:
    :param SORT:
    :param ManualSplit - Manual split for training and validating data
    :return:
    '''
    df = df.sort(df.id.asc())
    if manual_split:
        dfp = df.toPandas()
        dfp = np.array_split(dfp, train_fold + test_fold)
        train = spark.createDataFrame(data=dfp[0].round(3))
        for i in range(1, train_fold):
            p = spark.createDataFrame(data=dfp[i].round(3))
            train = train.union(p)
        test = spark.createDataFrame(data=dfp[-1].round(3))
        for j in range(-2, -test_fold - 1, -1):
            q = spark.createDataFrame(data=dfp[j].round(3))
            test = test.union(q)
        # Sorting
        test = test.sort(test.id.asc())
        train = train.sort(train.id.asc())
    else:
        train, test = df.randomSplit([0.7, 0.3], seed=random_seed)
    # print("We have %d training examples and %d test examples. \n" % (train.count(), test.count()))
    return train, test


def validate(df, random_seed):
    train, test = df.randomSplit([0.2, 0.8], seed=random_seed)
    return train, test


def complete_processing(spark, path):
    df = initial_processing(spark=spark, path_to_csv=path)
    # df = features_from_OHLC(spark=spark, spark_df=df)
    df = calc_profit(df=df)
    # df = calc_ti(spark, df)
    # return df.drop('High', 'Low', 'Close', 'Open')
    return df


def simple_processing(spark, path):
    df = initial_processing(spark=spark, path_to_csv=path)
    df = calc_profit2(df=df)
    df = df.select(
        [col(c).cast('float') for c in df.columns])
    return df
