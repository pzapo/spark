from io import StringIO

import boto3


def init_s3():
    # Let's use Amazon S3
    s3 = boto3.resource('s3')
    return s3


def save_on_s3(df, filename, bucket):
    test = df.toPandas()
    csv_buffer = StringIO()
    test.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, filename + '.csv').put(Body=csv_buffer.getvalue())
