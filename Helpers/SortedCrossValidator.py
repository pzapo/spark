import numpy as np
from pyspark.sql.functions import rand
from pyspark.ml.tuning import CrossValidatorModel, CrossValidator


class CrossValidatorSorted(CrossValidator):

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels

        dfp = df.toPandas()
        dfp = np.array_split(dfp, nFolds)

        train = self.spark.createDataFrame(data=dfp[0].round(3))
        for i in range(1, len(dfp) - 1):
            p = self.spark.createDataFrame(data=dfp[i].round(3))
            train = train.union(p)
        validation = self.spark.createDataFrame(data=dfp[-1].round(3))
        validation = validation.sort(validation.id.asc())
        train = train.sort(train.id.asc())


        for i in range(nFolds):
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition)
            validation = validation.sort(validation.id.asc())
            validation.show()
            train = df.filter(~condition)
            train = train.sort(train.id.asc())
            train.show()
            models = est.fit(train, epm)
            for j in range(numModels):
                model = models[j]
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation, epm[j]))
                metrics[j] += metric / nFolds

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics))