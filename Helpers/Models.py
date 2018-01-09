from collections import OrderedDict

from pyspark.ml.classification import LinearSVC, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder


def getSVMwithGrid(max_Iter=[10], reg_Param=[0.5], aggregation_Depth=[2], threshold_param=[0.5]):
    classifier = LinearSVC(labelCol='Profit', featuresCol="features")
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxIter, max_Iter) \
        .addGrid(classifier.threshold, threshold_param) \
        .addGrid(classifier.aggregationDepth, aggregation_Depth) \
        .addGrid(classifier.regParam, reg_Param).build()
    return (classifier, paramGrid)


def get_best_svm(model):
    print(model.getRegParam())
    # print(best_classifier.coefficients)
    print(model.getRegParam())
    print(model.getMaxIter())
    print(model.getAggregationDepth())
    print(model.getThreshold())
    print("Coefs")
    print(model.intercept)
    for coef in model.coefficients:
        print("{},".format(round(coef, 3)))
    model.write().overwrite().save("./Models/SVM_MI_{}_RP_{}_AG_{}".format(model.getMaxIter(),
                                                                           model.getRegParam(),
                                                                           model.getAggregationDepth()))


def getRandomForestwithGrid(num_Trees=25, max_Bins=500, max_Depth_Range=[8], min_infoGain=[0],
                            min_InstancesPerNode=[1]):
    classifier = RandomForestClassifier(
        labelCol='Profit', featuresCol="features", numTrees=num_Trees, maxBins=max_Bins, cacheNodeIds=True,
        maxMemoryInMB=4096)
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, max_Depth_Range) \
        .addGrid(classifier.minInfoGain, min_infoGain) \
        .addGrid(classifier.impurity, ['entropy', 'gini']) \
        .addGrid(classifier.minInstancesPerNode, min_InstancesPerNode).build()
    return classifier, paramGrid


def getDecisonTreewithGrid(max_Bins=200, max_Depth_Range=[8], min_infoGain=[0], min_InstancesPerNode=[1]):
    classifier = DecisionTreeClassifier(
        labelCol='Profit', featuresCol="features", maxBins=max_Bins, cacheNodeIds=True, maxMemoryInMB=4096)
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, max_Depth_Range) \
        .addGrid(classifier.minInfoGain, min_infoGain) \
        .addGrid(classifier.impurity, ['entropy', 'gini']) \
        .addGrid(classifier.minInstancesPerNode, min_InstancesPerNode).build()

    return classifier, paramGrid


def tree_feature_importances(best_tree, featuresCols):
    final_features = best_tree.featureImportances
    # Feature importance
    feature_dict = {}
    for feature, importance in zip(featuresCols, final_features):
        feature_dict[feature] = importance

    feature_dict = OrderedDict(sorted(feature_dict.items(), key=lambda t: t[1], reverse=True))

    i = 1
    for feature, importance in feature_dict.items():
        print("{} ; {} ; {}".format(i, feature, round(importance, 3)))
        i += 1


def best_tree_par(best_tree):
    # Max depth
    print("Maximal depth is " + str(best_tree.getMaxDepth()))
    max_depth = best_tree.getMaxDepth()

    # Min infoGain
    print("Minimal info gain is " + str(
        best_tree.getMinInfoGain()))
    minInfoGain = best_tree.getMinInfoGain()

    # Min instances
    print("Minimal instances per node is " + str(
        best_tree.getMinInstancesPerNode()))
    min_instancesPerNode = best_tree.getMinInstancesPerNode()

    # Min instances
    print("Impurity is " + str(
        best_tree.getImpurity()))
    impurity = best_tree.getImpurity()

    # Min instances
    print("MaxBins is " + str(
        best_tree.getMaxBins()))
    max_Bins = best_tree.getMaxBins()

    best_tree.write().overwrite().save("./Models/DD_MD_{}_MIG_{}_MIPN_{}_I_{}_MB_{}".format(max_depth,
                                                                                            minInfoGain,
                                                                                            min_instancesPerNode,
                                                                                            impurity,
                                                                                            max_Bins))
    
    return (max_depth, min_instancesPerNode, minInfoGain, impurity)
