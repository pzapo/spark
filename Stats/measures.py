from pyspark.mllib.evaluation import MulticlassMetrics


def calc_metrics(df, simple_mode=True):
    rdd = df.select("prediction", "Profit").rdd
    metrics = MulticlassMetrics(rdd)
    metrics_dict = {}
    cm = metrics.confusionMatrix().toArray()


    TP = cm[0][0]
    print("TP IS " + str(TP))

    TN = cm[1][1]
    print("TN IS " + str(TN))

    FP = cm[0][1]
    print("FP IS " + str(FP))

    FN = cm[1][0]
    print("FN IS " + str(FN))

    accuracy = (TP + TN) / cm.sum()
    sensitivity = (TP) / (TP + FN)
    specificity = (TN) / (TN + FP)
    precision = (TP) / (TP + FP)
    npv = (TN) / (TN + FN)

    # Overall statistics
    metrics_dict['accuracy'] = accuracy

    metrics_dict['sensitivity'] = sensitivity

    metrics_dict['specificity'] = specificity

    metrics_dict['precision'] = precision

    metrics_dict['npv'] = npv

    # print(metrics_dict)
    # print("Summary Stats")
    print(metrics.confusionMatrix())
    metrics_dict['confusionMatrix'] = metrics.confusionMatrix()
    # print("Accuracy = %.4f" % precision)
    # print("Recall = %.4f" % recall)
    # print("F1 Score = %.4f" % f1Score)

    # print("accuracy ", accuracy)
    # print("sensitivity ", sensitivity)
    # print("specificity ", specificity)
    # print("precision ", precision)

    print(
        "{},{},{},{},{}".format(round(accuracy, 3), round(sensitivity, 3), round(specificity, 3), round(precision, 3),
        round(npv, 3)))
    # print("sensitivity ", sensitivity)
    # print("specificity ", specificity)
    # print("precision ", precision)

    if not simple_mode:
        # Statistics by class
        labels = rdd.map(lambda lp: lp.Profit).distinct().collect()

        for label in sorted(labels):
            print("Class %s accuracy = %.4f" % (label, metrics.precision(label)))
            print("Class %s sensitivity = %.4f" % (label, metrics.recall(label)))
            print("Class %s F1 Measure = %.4f" % (label, metrics.fMeasure(label, beta=1.0)))
        print("\n")

        # Weighted stats
        weightedRecall = metrics.weightedRecall
        print("Weighted sensitivity = %.4f" % weightedRecall)
        metrics_dict['weightedRecall'] = weightedRecall

        weightedPrecision = metrics.weightedPrecision
        print("Weighted precision = %.4f" % weightedPrecision)
        metrics_dict['weightedPrecision'] = weightedPrecision

        # weightedFMeasure = metrics.weightedFMeasure()
        # print("Weighted F(1) Score = %.4f" % weightedFMeasure)
        # metrics_dict['weightedFMeasure'] = weightedFMeasure
        #
        # weightedFMeasure = metrics.weightedFMeasure(beta=0.5)
        # print("Weighted F(1) Score = %.4f" % weightedFMeasure)
        # metrics_dict['weightedFMeasure_beta'] = weightedFMeasure
        #
        # weightedFalsePositiveRate = metrics.weightedFalsePositiveRate
        # print("Weighted F(1) Score = %.4f" % weightedFalsePositiveRate)
        # metrics_dict['weightedFalsePositiveRate'] = weightedFalsePositiveRate
        print("\n")
    return metrics_dict


def get_metrics(df, lower_bound, upper_bound=1.0):
    rdd = df.select("prediction", "Profit").rdd
    metrics = MulticlassMetrics(rdd)
    metrics_dict = {}
    cm = metrics.confusionMatrix().toArray()

    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    accuracy = (TP + TN) / cm.sum()
    if accuracy < lower_bound or accuracy > upper_bound:
        return None
    sensitivity = (TP) / (TP + FN)
    specificity = (TN) / (TN + FP)
    precision = (TP) / (TP + FP)
    npv = (TN) / (TN + FN)

    # Overall statistics
    metrics_dict['accuracy'] = accuracy

    metrics_dict['sensitivity'] = sensitivity

    metrics_dict['specificity'] = specificity

    metrics_dict['precision'] = precision

    metrics_dict['npv'] = npv

    # print("Summary Stats")
    # print(metrics.confusionMatrix())
    metrics_dict['confusionMatrix'] = metrics.confusionMatrix()

    print(
        "{},{},{},{},{}".format(round(accuracy, 3), round(sensitivity, 3), round(specificity, 3), round(precision, 3),
                                round(npv, 3)))

    return metrics_dict
