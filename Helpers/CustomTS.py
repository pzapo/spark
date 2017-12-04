import numpy as np

from pyspark.ml.tuning import TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql.functions import rand


class TrainValidationSplitSorted(TrainValidationSplit):
    chunks = 0
    spark = None

    def __init__(self, chunks, spark):
        self.chunks = chunks
        self.spark = spark
        super().__init__()

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        tRatio = self.getOrDefault(self.trainRatio)
        seed = self.getOrDefault(self.seed)
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels
        condition = (df[randCol] >= tRatio)

        dfp = df.toPandas()
        dfp = np.array_split(dfp, self.chunks)
        train = self.spark.createDataFrame(data=dfp[0].round(3))
        for i in range(1, len(dfp) - 1):
            p = self.spark.createDataFrame(data=dfp[i].round(3))
            train = train.union(p)
        validation = self.spark.createDataFrame(data=dfp[-1].round(3))
        validation = validation.sort(validation.id.asc())
        train = train.sort(train.id.asc())

        train.show(20)
        print('#######################################################################')
        validation.show(20)
        models = est.fit(train, epm)
        for j in range(numModels):
            model = models[j]
            metric = eva.evaluate(model.transform(validation, epm[j]))
            metrics[j] += metric
        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(TrainValidationSplitModel(bestModel, metrics))
