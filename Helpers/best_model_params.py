from pyspark.ml.classification import RandomForestClassificationModel

RandomForestClassificationModel.getMaxDepth = (
    lambda self: self._java_obj.getMaxDepth())

RandomForestClassificationModel.getMinInstancesPerNode = (
    lambda self: self._java_obj.getMinInstancesPerNode())
