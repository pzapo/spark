from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel, LinearSVCModel

RandomForestClassificationModel.getMaxDepth = (
    lambda self: self._java_obj.getMaxDepth())

RandomForestClassificationModel.getMinInstancesPerNode = (
    lambda self: self._java_obj.getMinInstancesPerNode())

RandomForestClassificationModel.getMinInfoGain = (
    lambda self: self._java_obj.getMinInfoGain())

RandomForestClassificationModel.getImpurity = (
    lambda self: self._java_obj.getImpurity())

DecisionTreeClassificationModel.getMaxDepth = (
    lambda self: self._java_obj.getMaxDepth())

DecisionTreeClassificationModel.getMinInstancesPerNode = (
    lambda self: self._java_obj.getMinInstancesPerNode())

DecisionTreeClassificationModel.getMinInfoGain = (
    lambda self: self._java_obj.getMinInfoGain())

DecisionTreeClassificationModel.getImpurity = (
    lambda self: self._java_obj.getImpurity())

LinearSVCModel.getRegParam = (
    lambda self: self._java_obj.getRegParam())

LinearSVCModel.getmaxIter = (
    lambda self: self._java_obj.getmaxIter())
