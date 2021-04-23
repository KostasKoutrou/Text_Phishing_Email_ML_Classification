from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler


#This class retrieves features from a DataFrame using the featurefunc feature retrieving
#function, then trains a classifier and evaluates its performance. It can also get
#a DataFrame which already has the features retrieved if there is one.
class Modeler:
    
    def __init__(self, DF=None, featurefunc=None, classifier=None, featuresDF=None):
        self.DF = DF
        self.featurefunc = featurefunc
        self.classifier = classifier
        self.featuresDF = featuresDF
    
    def retrievefeatures(self):
        #This function gets the DF and the featurefunc and creates the featuresDF
        self.featuresDF = self.featurefunc(self.DF)
    
    def splitDF(self):
        self.dftrain, self.dftest = self.featuresDF.randomSplit([0.80, 0.20])
    
    def trainModel(self):
        self.model = self.classifier.fit(self.dftrain)
    
    def testModel(self):
        predictions = self.model.transform(self.dftest)
#        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
#        areaUnderROC = evaluator.evaluate(predictions)
#        predictionRDD = predictions.select(['label', 'prediction']).rdd.map(lambda line: (line[1], line[0]))
#        
#        binmetrics = BinaryClassificationMetrics(predictionRDD)
#        metrics = MulticlassMetrics(predictionRDD)
#        
#        self.results = {'predictions':predictions,
#                        'areaUnderROC':binmetrics.areaUnderROC,
#                        'areaUnderPR':binmetrics.areaUnderPR,
#                        'confusionMatrix':metrics.confusionMatrix().toArray(),
#                        'accuracy':metrics.accuracy,
#                        'precision':metrics.precision(),
#                        'recall':metrics.recall(),
#                        'f1measure':metrics.fMeasure()}
#        return self.results
        evaluator = BinaryClassificationEvaluator()
        print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
        cm = predictions.select('label','prediction')
        print("Accuracy: " + str(cm.filter(cm.label == cm.prediction).count() / cm.count()))
    
    def performance(self):
        pass



#This is a function equal to the Class above. It retrieves features using the featurerfunc
#feature retrieving function, trains a classifier and evaluates its performance.
def modeling(DF, featurefunc, classifier, featuresDF=None):
    if(not featuresDF):
        print('DF with features not provided, building with using ' + featurefunc.__name__ + '...')
        featuresDF = featurefunc(DF)
    else:
        print('DF with features provided, moving on next step...')
    
    dftrain, dftest = featuresDF.randomSplit([0.80, 0.20])
    
    print('Training model with classifier ' + str(classifier) + '...')
    model = classifier.fit(dftrain)
    
    print('Evaluating model...')
    predictions = model.transform(dftest)
    
    print('Calculating performance metrics...')
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
#    areaUnderROC = evaluator.evaluate(predictions)
#    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / predictions.count()
    results = {'model':model,
               'predictions':predictions,
#               'areaUnderROC':areaUnderROC,
#               'accuracy':accuracy
               }
#    predictionRDD = predictions.select(['label', 'prediction']).rdd.map(lambda line: (line[1], line[0]))
#        
#    binmetrics = BinaryClassificationMetrics(predictionRDD)
#    metrics = MulticlassMetrics(predictionRDD)
#    
#    results = {'predictions':predictions,
#               'areaUnderROC':binmetrics.areaUnderROC,
#               'areaUnderPR':binmetrics.areaUnderPR,
#               'confusionMatrix':metrics.confusionMatrix().toArray(),
#               'accuracy':metrics.accuracy,
#               'precision':metrics.precision(),
#               'recall':metrics.recall(),
#               'f1measure':metrics.fMeasure()}
    
    return results




#This class takes a DataFrame, the Transformation Stages to retrieve features
#and a classifier, and then trains the classifier and evaluates its performance.
class pipemodeler:
    def __init__(self, DF, featurestages, classifier, classifiergrid=None):
        self.DF = DF
        self.featurestages = featurestages
        self.classifier = classifier
        self.classifiergrid = classifiergrid
    
    def train(self):
        print('Building stages...')
        stages = []
        if(type(self.featurestages)!=list):
            self.featurestages=[self.featurestages]
        stages += self.featurestages
        
        
        #In case there is word2vec which has negative features, scale the features
        #to nonnegative values because naive bayes requires that
        if(('Word2Vec' in str(stages)) and ('NaiveBayes' in str(self.classifier))):
            print('Word2Vec and NaiveBayes detected, scaling to nonnegative [0.0,1.0]')
            stages[-1].setOutputCol('prefeatures')
            scaler  = MinMaxScaler(inputCol='prefeatures', outputCol='features')
            stages = stages + [scaler]
        
        
        stages += [self.classifier]
        self.pipeline = Pipeline(stages = stages)
        
        print('Using the following stages: ' + str(self.pipeline.getStages()))
        print('Training model...')
        if(self.classifiergrid == None):
            print('Training without a Parameter Grid...')
            dftrain, dftest = self.DF.randomSplit([0.80, 0.20])
            model = self.pipeline.fit(dftrain)
            self.predictions = model.transform(dftest)
            self.model=model
        else:
            # print('Training with a Parameter Grid...')
            # tvs = TrainValidationSplit(estimator=self.pipeline,
            #                             estimatorParamMaps=self.classifiergrid,
            #                             evaluator=BinaryClassificationEvaluator(),
            #                             parallelism=4s,
            #                             trainRatio=0.7)
            # dftrain, dftest = self.DF.randomSplit([0.70, 0.30])
            # model = tvs.fit(dftrain)
            print('Cross Validation Hyperparamter Tunning...')
            cv = CrossValidator(estimator=self.pipeline,
                                estimatorParamMaps=self.classifiergrid,
                                evaluator=BinaryClassificationEvaluator(),
                                parallelism=4,
                                numFolds=5)
            dftrain, dftest = self.DF.randomSplit([0.70, 0.30])
            model = cv.fit(dftrain)
            self.predictions = model.transform(dftest)
            self.model=model
            

    def performancerdd(self):
        self.calculator = 'RDDs'
        print('Calculating performance metrics using RDDs...')
        predictionRDD = self.predictions.select(['label','prediction']).rdd.map(lambda line: (line[1],line[0]))
        
        binmetrics = BinaryClassificationMetrics(predictionRDD)
        metrics = MulticlassMetrics(predictionRDD)
        
        self.areaUnderROC = binmetrics.areaUnderROC
        self.areaUnderPR = binmetrics.areaUnderPR
        self.confusionMatrix = metrics.confusionMatrix().toArray()
        self.accuracy = metrics.accuracy
        self.precision = metrics.precision()
        self.recall = metrics.recall()
        self.f1measure = metrics.fMeasure()
        self.falsePositive = metrics.falsePositiveRate(1.0)
        self.falseNegative = metrics.falsePositiveRate(0.0)
    
    def performance(self):
        self.calculator = 'Nothing'
        print('Calculating performance metrics using nothing...')
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
        self.areaUnderROC = evaluator.evaluate(self.predictions)
        preds = self.predictions
        fp = preds.filter(preds.label<preds.prediction).count()
        fn = preds.filter(preds.label>preds.prediction).count()
        tp = preds.filter(preds.label==1.0).filter(preds.prediction==1.0).count()
        tn = preds.filter(preds.label==0.0).filter(preds.prediction==0.0).count()
        total = fp + fn + tp + tn
        self.confusionMatrix = [[tn,fn],[fp,tp]]
        self.accuracy = (tp+tn)/total
        if(tp+fp):
            self.precision = tp / (tp + fp)
        else:
            self.precision = 0
        if(tp+fn):
            self.recall = tp / (tp + fn)
        else:
            self.recall = 0
        if(self.precision + self.recall):
            self.f1measure = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1measure = 0
        if(fp+tn):
            self.falsePositive = fp / (fp + tn)
        else:
            self.falsePositive = 0
        if(fn+tp):
            self.falseNegative = fn / (fn + tp)
        else:
            self.falseNegative = 0
        
    
    def printperformance(self):
        print('Stages: ' + str(self.pipeline.getStages()))
        print('Performance calculated using ' + self.calculator)
        print('areaUnderROC = ' + str(self.areaUnderROC))
#        print('areaUnderPR = ' + str(self.areaUnderPR))
        print('confusionMatrix:')
        print(self.confusionMatrix)
        print('accuracy = ' + str(self.accuracy))
        print('precision = ' + str(self.precision))
        print('recall = ' + str(self.recall))
        print('f1measure = ' + str(self.f1measure))
        print('falsePositive = ' + str(self.falsePositive))
        print('falseNegative = ' + str(self.falseNegative))
        



#This function gets a DataFrame and a pipeline which include at least a classifier
#and then returns the model and a DataFrame with predictions.
def pipemodeling(DF, pipeline):
    dftrain, dftest = DF.randomSplit([0.80, 0.20])
    model = pipeline.fit(dftrain)
    predictions = model.transform(dftest)
    results = {'model':model,
               'predictions':predictions}
    return results    



#This function works as a follow-up of the previous one. It takes a DataFrame with
#predictions and returns the performance of the classifier.
def performance(predictions):
    predictionRDD = predictions.select(['label', 'prediction']).rdd.map(lambda line: (line[1], line[0]))
        
    binmetrics = BinaryClassificationMetrics(predictionRDD)
    metrics = MulticlassMetrics(predictionRDD)
    
    results = {'predictions':predictions,
               'areaUnderROC':binmetrics.areaUnderROC,
               'areaUnderPR':binmetrics.areaUnderPR,
               'confusionMatrix':metrics.confusionMatrix().toArray(),
               'accuracy':metrics.accuracy,
               'precision':metrics.precision(),
               'recall':metrics.recall(),
               'f1measure':metrics.fMeasure()}
    
    return results