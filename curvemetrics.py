from pyspark.mllib.evaluation import BinaryClassificationMetrics
import matplotlib.pyplot as plt

# Scala version implements .roc() and .pr()
# Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
# Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)


def plotCurves(classlist):
    plt.figure(figsize=(9,5),dpi=100)
    plt.title('ROC Curve')
    plt.xlabel('1-Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    
    for result in classlist:
        try:
            preds = result.predictions.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
        except:
            continue
        roc = CurveMetrics(preds).get_curve('roc')
        x_val = [x[0] for x in roc]
        y_val = [x[1] for x in roc]
    
        plt.plot(x_val, y_val, label=str(result.pipeline.getStages()[-1])[:-13])
    
    plt.legend(loc='best')
    plt.show()
#    plt.close()