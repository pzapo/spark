from pyspark.mllib.evaluation import MulticlassMetrics


def calc_metrics(df):
    rdd = df.select("prediction", "Profit").rdd
    metrics = MulticlassMetrics(rdd)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()

    print("Summary Stats")
    print(metrics.confusionMatrix())
    print("Accuracy = %.4f" % precision)
    print("Recall = %.4f" % recall)
    print("F1 Score = %.4f" % f1Score)
    print("\n")

    # Statistics by class
    labels = rdd.map(lambda lp: lp.Profit).distinct().collect()

    for label in sorted(labels):
        print("Class %s precision = %.4f" % (label, metrics.precision(label)))
        print("Class %s recall = %.4f" % (label, metrics.recall(label)))
        print("Class %s F1 Measure = %.4f" % (label, metrics.fMeasure(label, beta=1.0)))
    print("\n")

    # Weighted stats
    print("Weighted recall = %.4f" % metrics.weightedRecall)
    print("Weighted precision = %.4f" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %.4f" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %.4f" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %.4f" % metrics.weightedFalsePositiveRate)

    return True


def best_model_prams(best_classifier):
    # Max depth
    print("Maximal depth is " + str(best_classifier.getMaxDepth()))
    max_depth = best_classifier.getMaxDepth()

    # Min instances
    print("Minimal instances per node is " + str(
        best_classifier.getMinInstancesPerNode()))
    min_instancesPerNode = best_classifier.getMinInstancesPerNode()

    return (max_depth, min_instancesPerNode)
