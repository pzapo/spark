import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id, udf, split
from pyspark.sql.types import IntegerType

from Helpers.generated_features import features_from_OHLC
from Helpers.technical_indicators import calc_ti
from Helpers.udf import profit_

profit_udf = udf(profit_, IntegerType())


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
    fresh_df = fresh_df.filter(fresh_df.Open != "null")
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
                                    F.lag(df['Close']).over(
                                        Window.orderBy("id")))
    df_daily_return = df_daily_return.filter(
        df_daily_return.prev_day_price.isNotNull())
    df_profit = df_daily_return.withColumn(
        'Profit', profit_udf(df_daily_return.Close,
                             df_daily_return.prev_day_price))

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


def train_test_split(spark, df, CHUNKS, SORT, ManualSplit, RANDOM_SEED):
    '''
    :param spark:
    :param df:
    :param CHUNKS:
    :param SORT:
    :param ManualSplit - Manual split for training and validating data
    :return:
    '''
    df = df.sort(df.id.asc())
    if ManualSplit:
        dfp = df.toPandas()
        dfp = np.array_split(dfp, CHUNKS)
        train = spark.createDataFrame(data=dfp[0].round(3))
        for i in range(1, len(dfp) - 1):
            p = spark.createDataFrame(data=dfp[i].round(3))
            train = train.union(p)
        test = spark.createDataFrame(data=dfp[-1].round(3))
    else:
        train, test = df.randomSplit([0.85, 0.15], seed=RANDOM_SEED)

    print("We have %d training examples and %d test examples." % (train.count(),
                                                                  test.count()))
    if SORT:
        test = test.sort(test.id.asc())
        train = train.sort(train.id.asc())
    return train, test


def complete_processing(spark, path):
    df = initial_processing(spark=spark, path_to_csv=path)
    df = features_from_OHLC(spark=spark, spark_df=df)
    df = calc_profit(df=df)
    df = calc_ti(spark, df)
    return df
