import numpy as np

from pyspark.ml.tuning import TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql.functions import rand


class TrainValidationSplitSorted(TrainValidationSplit):
    chunks = 0
    spark = None

    def __init__(self, spark=None, estimator=None, estimatorParamMaps=None, evaluator=None,
                 trainRatio=0.75,
                 seed=None, train_fold=None,test_fold=None):
        super().__init__(estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=evaluator,
                         trainRatio=trainRatio, seed=seed)
        self.spark = spark
        self.test_fold = test_fold
        self.train_fold = train_fold

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
        train_fold = self.train_fold
        test_fold = self.test_fold

        df = df.sort(df.id.asc())
        dfp = df.toPandas()
        dfp = np.array_split(dfp, train_fold + test_fold)
        train = self.spark.createDataFrame(data=dfp[0].round(3))
        for i in range(1, train_fold):
            p = self.spark.createDataFrame(data=dfp[i].round(3))
            train = train.union(p)
        validation = self.spark.createDataFrame(data=dfp[-1].round(3))
        for j in range(-2, -test_fold-1, -1):
            q = self.spark.createDataFrame(data=dfp[j].round(3))
            validation = validation.union(q)
        validation = validation.sort(validation.id.asc())
        train = train.sort(train.id.asc())

        # train.select(train.id).show(14000)
        # print('#######################################################################')
        # validation.select(validation.id).show(14000)

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
