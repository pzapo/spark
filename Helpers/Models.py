from pyspark.ml.classification import LinearSVC, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder


def getSVMwithGrid(max_Iter=[10], reg_Param=[0.5], aggregation_Depth = [2]):
    classifier = LinearSVC(labelCol='Profit', featuresCol="features")
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxIter, max_Iter) \
        .addGrid(classifier.aggregationDepth, aggregation_Depth) \
        .addGrid(classifier.regParam, reg_Param).build()
    return (classifier, paramGrid)


def getRandomForestwithGrid(num_Trees=25, max_Bins=500, max_Depth_Range=[8], min_infoGain=[0],
                            min_InstancesPerNode=[1]):
    classifier = RandomForestClassifier(
        labelCol='Profit', featuresCol="features", numTrees=num_Trees, maxBins=max_Bins,cacheNodeIds=True,maxMemoryInMB=4096)
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, max_Depth_Range) \
        .addGrid(classifier.minInfoGain, min_infoGain) \
        .addGrid(classifier.impurity, ['entropy','gini']) \
        .addGrid(classifier.minInstancesPerNode, min_InstancesPerNode).build()
    return classifier, paramGrid


def getDecisonTreewithGrid(max_Bins=200, max_Depth_Range=[8], min_infoGain=[0], min_InstancesPerNode=[1]):
    classifier = DecisionTreeClassifier(
        labelCol='Profit', featuresCol="features", maxBins=max_Bins, cacheNodeIds=True,maxMemoryInMB=4096)
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, max_Depth_Range) \
        .addGrid(classifier.minInfoGain, min_infoGain) \
        .addGrid(classifier.impurity, ['entropy', 'gini']) \
        .addGrid(classifier.minInstancesPerNode, min_InstancesPerNode).build()

    return classifier, paramGrid
