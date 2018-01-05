from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel

RandomForestClassificationModel.getMaxDepth = (
    lambda self: self._java_obj.getMaxDepth())

RandomForestClassificationModel.getMinInstancesPerNode = (
    lambda self: self._java_obj.getMinInstancesPerNode())

RandomForestClassificationModel.getMinInfoGain = (
    lambda self: self._java_obj.getMinInfoGain())

DecisionTreeClassificationModel.getMaxDepth = (
    lambda self: self._java_obj.getMaxDepth())

DecisionTreeClassificationModel.getMinInstancesPerNode = (
    lambda self: self._java_obj.getMinInstancesPerNode())

DecisionTreeClassificationModel.getMinInfoGain = (
    lambda self: self._java_obj.getMinInfoGain())