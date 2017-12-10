from pyspark.mllib.evaluation import MulticlassMetrics


def calc_metrics(df, simple_mode=True):
    rdd = df.select("prediction", "Profit").rdd
    metrics = MulticlassMetrics(rdd)
    metrics_dict = {}
    # Overall statistics
    precision = metrics.precision()
    metrics_dict['precision'] = precision
    recall = metrics.recall()
    metrics_dict['recall'] = recall
    f1Score = metrics.fMeasure()
    metrics_dict['f1Score'] = f1Score

    print("Summary Stats")
    print(metrics.confusionMatrix())
    metrics_dict['confusionMatrix'] = metrics.confusionMatrix()
    print("Accuracy = %.4f" % precision)
    print("Recall = %.4f" % recall)
    print("F1 Score = %.4f" % f1Score)

    if not simple_mode:
        # Statistics by class
        labels = rdd.map(lambda lp: lp.Profit).distinct().collect()

        for label in sorted(labels):
            print("Class %s precision = %.4f" % (label, metrics.precision(label)))
            print("Class %s recall = %.4f" % (label, metrics.recall(label)))
            print("Class %s F1 Measure = %.4f" % (label, metrics.fMeasure(label, beta=1.0)))
        print("\n")

        # Weighted stats
        weightedRecall = metrics.weightedRecall
        print("Weighted recall = %.4f" % weightedRecall)
        metrics_dict['weightedRecall'] = weightedRecall

        weightedPrecision = metrics.weightedPrecision
        print("Weighted precision = %.4f" % weightedPrecision)
        metrics_dict['weightedPrecision'] = weightedPrecision

        weightedFMeasure = metrics.weightedFMeasure()
        print("Weighted F(1) Score = %.4f" % weightedFMeasure)
        metrics_dict['weightedFMeasure'] = weightedFMeasure

        weightedFMeasure = metrics.weightedFMeasure(beta=0.5)
        print("Weighted F(1) Score = %.4f" % weightedFMeasure)
        metrics_dict['weightedFMeasure_beta'] = weightedFMeasure

        weightedFalsePositiveRate = metrics.weightedFalsePositiveRate
        print("Weighted F(1) Score = %.4f" % weightedFalsePositiveRate)
        metrics_dict['weightedFalsePositiveRate'] = weightedFalsePositiveRate
        print("\n")
    return metrics_dict


def best_model_prams(best_classifier):
    # Max depth
    print("Maximal depth is " + str(best_classifier.getMaxDepth()))
    max_depth = best_classifier.getMaxDepth()

    # Min instances
    print("Minimal instances per node is " + str(
        best_classifier.getMinInstancesPerNode()))
    min_instancesPerNode = best_classifier.getMinInstancesPerNode()

    return (max_depth, min_instancesPerNode)
