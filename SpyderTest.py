import mailbox
import utils, MLModeler
import os
import re
import time
import curvemetrics
import sparknlp
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pyspark import keyword_only
# from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, JavaMLReadable, JavaMLWritable
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier\
, LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier\
, LinearSVC, NaiveBayes
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.feature import HashingTF, IDF, Word2Vec,\
OneHotEncoderEstimator, StringIndexer, VectorAssembler, ChiSqSelector
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline, Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import explode, udf, col
from functools import reduce
from pyspark.ml.linalg import VectorUDT, DenseVector, SparseVector
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.embeddings import *
from pyspark.sql import SparkSession


from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml.param.shared import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 


# spark = SparkSession.builder \
#     .appName("Spark NLP")\
#     .master("local[4]")\
#     .config("spark.driver.memory","16G")\
#     .config("spark.executor.memory","10G")\
#     .config("spark.driver.maxResultSize", "0") \
#     .config("spark.kryoserializer.buffer.max", "1000M")\
#     .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.1")\
#     .getOrCreate()
spark = sparknlp.start()
spark.sparkContext.setLogLevel("ERROR")



start_time = time.time()



#Returns the MD5 hash of the stemmed list of words
class MD5hash:
    def get_feat(self, msg):
        pass
    def get_name(self):
        return 'MD5hash'


#Returns the year that the email was sent in.
class year:
    def get_feat(self, msg):
        date = None
        if msg['Date']: date = msg['Date']
        elif msg['date']: date  = msg['date']
        else: year = 'no_date_field'
        if date:
            regex = r"([1-2][0-9]{3})"
            year = re.findall(regex, date)
            year = year[0] if year else 'yearnotfound_in_date_field'
        return year
    def get_name(self):
        return 'year'


#Returns a list of URLs that are in the email.
class URLs:
    def get_feat(self, msg):
        emailtext = msg.as_string()
        emailtext = emailtext.replace('=\n','')
#        soup = BeautifulSoup(emailtext, 'lxml')
#        urls = [link.get('href') for link in soup.find_all('a')]
#        for a in soup.find_all('a'):
#            a.decompose()
#        emailtext = str(soup.contents[0])
        regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        urls = re.findall(regex, emailtext)
        #Appending an empty string to declare urls as a strings list in case
        #it is an empty string, for DataFrame consistency
        urls.append('')
        return urls
    def get_name(self):
        return 'urls'


#Returns the maximum number of dots found at the URLs of the email.
class ndots:
    def get_feat(self, msg):
        finder = URLs()
        urls = finder.get_feat(msg)
        ndots = 0
        for url in urls:
            temp_dots = url.count(".")
            if(temp_dots>ndots): ndots = temp_dots
        return ndots
    def get_name(self):
        return 'ndots'


#Returns the number of specified port numbers in the URLs of the email.
class nports:
    def get_feat(self, msg):
        finder = URLs()
        urls = finder.get_feat(msg)
        nports = 0
        for url in urls:
            try:
                parsedurl = urlparse(url)
                if(parsedurl.port):
                    nports+=1
            except:
                pass
        return nports
    def get_name(self):
        return 'nports'


#Returns the total number of email recipients.
class nrecs:
    def get_feat(self, msg):
        rectypes = ['To', 'Cc', 'Bcc']
        reclist=[]
        for rectype in rectypes:
            temp = msg[rectype]
            if(temp):
                temp_recs = re.findall(r'[\w\.-]+@[\w\.-]+', temp)
                reclist = reclist + temp_recs
        reclist = list(dict.fromkeys(reclist))
        reclistlen = len(reclist)
        if(reclistlen < 0): print(reclistlen)
        return reclistlen
    def get_name(self):
        return 'nrecs'


#Boolean: Checks whether the sender domain is the same as the URLs' domains in the email.
class checkdomains:
    def get_feat(self, msg):
        checkdomains = 1
        if(msg['From']):
            tempdom = msg['From']
        elif(msg['Return-Path']):
            tempdom = msg['Return-Path']
        else:
            tempdom = ""
        sender = re.findall(r'@[\w.\-]+', tempdom)
        if(sender):
            sender_domain = sender[0][1:]
            urls = URLs().get_feat(msg)
            for url in urls:
                if(url and (sender_domain not in url)):
                    checkdomains = 0
        else:
            checkdomains = 0
        return checkdomains
    def get_name(self):
        return 'checkdomains'


#Returns the amount of URLs that the email has.
class NURLs:
    def get_feat(self, msg):
        finder = URLs()
        urls = finder.get_feat(msg)
#        Because the class URLs appends an empty string, substitute 1.
        return len(urls)-1
    def get_name(self):
        return 'nurls'


#Returns the Content Transfer Encoding of the email.
class encoding:
    def get_feat(self, msg):
        enc = msg["Content-Transfer-Encoding"]
        if(not enc):
            enc = 'none'
        enc = enc.lower()
        enc = re.findall(r'[\w-]+', enc)[0]
#        enc.replace("\n","")
#        enc.replace("\r","")
#        enc.replace(" ","")
        return enc
    def get_name(self):
        return 'encoding'


#Returns an integer  representing a different a encoding.
class nencoding:
    def get_feat(self, msg):
        encodings = {
                "quoted-printable":1,
                "none":2,
                "base64":3,
                "utf-8":4,
                "8bit":5,
                "binary":6,
                "7bit":7,
                }
        enc = msg["Content-Transfer-Encoding"]
        if(not enc):
            enc = 'none'
        enc = enc.lower()
        enc = re.findall(r'[\w-]+', enc)[0]
        if(enc in encodings):
            nenc = encodings[enc]
        else:
            nenc = 8
        return nenc
    def get_name(self):
        return "nencoding"


#Returns the amount of parts that the email has.
class nparts:
    def get_feat(self, msg):
        parts = 0
        for part in msg.walk():
            content = part.get_content_type()
            if('multipart' not in content):
                parts += 1
        return parts
    def get_name(self):
        return 'nparts'


#Returns whether the email has an HTML part or not.
class hasHTML:
    def get_feat(self, msg):
        hashtml = 0
        for part in msg.walk():
            if(part.get_content_type() == 'text/html'):
                hashtml = 1
        return hashtml
    def get_name(self):
        return 'hasHTML'


#Returns the amount of attachments (not inline) that are in the email.
class attachments:
    def get_feat(self,msg):
        counter = 0
        for part in msg.walk():
            disp = part.get_content_disposition()
            if(disp=='attachment'):
                counter += 1
        return counter
    def get_name(self):
        return 'attachments'


#Returns the amount of "bad words" (the ones shown in the list in the function)
#that are in the email.
class badwords:
    def get_feat(self,msg):
        words = ['link','click','confirm','user','customer',
                 'client','suspend','restrict','verify','protect']
        clean_text = utils.get_clean_text(msg).lower()
        counter = 0
        for word in words:
            counter += clean_text.count(word)
        return counter
    def get_name(self):
        return 'badwords'


#Returns the total size of the email (including the size of any attached/inline file.)
class size:
    def get_feat(self,msg):
        return len(msg.as_string())
    def get_name(self):
        return 'size'


#Returns the amount of IP URLs that are in the email.
class ipurls:
    def get_feat(self,msg):
        ipregex = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        finder = URLs()
        urls = finder.get_feat(msg)
        totalips = []
        for url in urls:
            ips = re.findall(ipregex, url)
            for ip in ips:
                if(ip not in totalips):
                    totalips.append(ip)
        return len(totalips)
    def get_name(self):
        return 'ipurls'


#Returns the amount of URLs that are different than the hyperlink text in the email.
class diffhref:
    def get_feat(self,msg):
        soup = BeautifulSoup(msg.as_string(), 'lxml')
        links = soup.find_all('a')
        counter = 0
        for link in links:
            url = link.get('href')
            urltext = link.text
            if(url != urltext): counter += 1
        return counter
    def get_name(self):
        return 'diffhref'


#Returns the amount of HTML Form blocks that are in the email.
class forms:
    def get_feat(self,msg):
        soup = BeautifulSoup(msg.as_string(), 'lxml')
        counter = len(soup.find_all('form'))
        return counter
    def get_name(self):
        return 'forms'

#Returns the amount of Javascript blocks that are in the email.
class scripts:
    def get_feat(self,msg):
        soup = BeautifulSoup(msg.as_string(),'lxml')
        counter = len(soup.find_all('script'))
        return counter
    def get_name(self):
        return 'scripts'



#Used to help the next function to operate properly.
def mbox_reader(stream):
    """Read a non-ascii message from mailbox"""
    data = stream.read()
#    text = data.decode(encoding="utf-8", errors="replace")
    text = data.decode(encoding="latin-1")
    return mailbox.mboxMessage(text)


#Takes a dataset filepath, whether it includes phishing emails, and a limit
#and creates a DataFrame with the email text as stemmed words and properties features.
def mboxText2DF(filepath, Phishy, limit=5000):
    print("Processing file: " + filepath)
    mbox = mailbox.mbox(filepath, factory=mbox_reader)
    #mbox = mailbox.mbox(filepath)
    email_index = []
    finders = [NURLs(), encoding(), nparts(), hasHTML(), attachments(),
               badwords(), ipurls(), diffhref(), forms(), scripts(),
               ndots(), nports(), nrecs(), checkdomains()]

    i = 1
    for message in mbox:
    #    input(str(i) + "ENTER FOR NEXT") #For testing
        if(not utils.is_empty(message)):
#            print("    NEW MESSAGE")
            
            email_clean_text = utils.get_clean_text(message)
            feats = [finder.get_feat(message) for finder in finders]
            email_index.append([i, Phishy, email_clean_text] + feats)
            
#            email_index.append((i, Phishy, email_clean_text))
    #        print(email_text) #For testing
    #        print(i)
            i += 1
            if i > limit: break
        else: print("EMPTY EMAIL - Moving to next email...")
    
    
#    emailDF = spark.createDataFrame(email_index,('id', 'label', 'emailText'))
    emailDF = spark.createDataFrame(email_index,['id', 'label', 'emailText'] + [finder.get_name() for finder in finders])
    emailDF = utils.textDF2setDF(emailDF, "emailText")
    emailDF = emailDF.drop('emailText', 'words', 'stopWremoved')
    return emailDF



#Creates a Dataframe with total size of 2*halflimit, using the DataSets provided
#at the specified directories.
def createDF(halflimit=2500):
    DFlist = []
    
    limitcounter = halflimit
    for root,dirs,files in os.walk("phishing_datasets"):
        pass
    for file in files:
        if(limitcounter>0):
            tempDF = mboxText2DF(root+"/"+file, Phishy=1.0, limit=limitcounter)
            DFlist.append(tempDF)
            limitcounter -= tempDF.count()
            
    limitcounter = halflimit
    for root,dirs,files in os.walk("legit_datasets"):
        pass
    for file in files:
        if(limitcounter>0):
            tempDF = mboxText2DF(root+"/"+file, Phishy=0.0, limit=limitcounter)
            DFlist.append(tempDF)
            limitcounter -= tempDF.count()
    
    print("---Total DF Created---")
    print(DFlist)
    DF = reduce(DataFrame.union, DFlist)
    DF = DF.drop('id')
    for item in DF.dtypes:
        if item[1] == 'bigint': DF = DF.withColumn(item[0], DF[item[0]].cast('int'))
        if item[1] == 'double': DF = DF.withColumn(item[0], DF[item[0]].cast('float'))
        
#    DF = DF.select('id','label','stemmed')
    return DF



#Takes a DataFrame and returns predictions based on TF-IDF features and
#the Logistic Regression Classifier.
def tfidf(DF,inputCol="stemmed"):
    #Create Evaluator Instance
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    
    #Create Pipelines stages Instaces
    tf1 = HashingTF(inputCol=inputCol, outputCol="rawFeatures", numFeatures=500)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2.0)
    lr = LogisticRegression(maxIter=20)
    basic_pipeline = Pipeline(stages=[tf1, idf, lr])
    
    #Split Dataset into Training and Testing subsets
    dftrain, dftest = DF.randomSplit([0.80, 0.20])
    
    #Create model by getting the Training set through the Pipeline
    model = basic_pipeline.fit(dftrain)
    
    #Make predictions for the Testing set and evaluate the model's performance
    predictions = model.transform(dftest)
#    score = evaluator.evaluate(predictions)
#    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / predictions.count()
    return {'TFIDFmodel':model,
            'predictions':predictions,
#            'areaUnderROC':score,
#            'accuracy':accuracy
            }



#Takes a DataFrame and returns predictions based on Word Embedding features and
#the Gradient Boosted Trees Classifier.
def embedding(DF, inputCol="stemmed"):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    word2vec = Word2Vec(inputCol = inputCol, outputCol = 'features')
    model = word2vec.fit(DF)
    resultDF = model.transform(DF)
    dftrain, dftest = resultDF.randomSplit([0.80, 0.20])
    gbt = GBTClassifier(maxDepth=5)
    model = gbt.fit(dftrain)
    predictions = model.transform(dftest)
    score = evaluator.evaluate(predictions)
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / predictions.count()
    return {'embeddingmodel':model,
            'predictions':predictions,
            'areaUnderROC':score,
            'accuracy':accuracy
            }



#Takes a DataFrame and returns a DataFrame with a column of TF-IDF Features.
def tfidffeats(DF, inputCol='stemmed'):
    #Takes stemmed text DF and returns TF-IDF Features
    tf = HashingTF(inputCol=inputCol, outputCol='rawFeatures', numFeatures=500)
    idf = IDF(inputCol='rawFeatures', outputCol='features', minDocFreq=2.0)
    pipeline = Pipeline(stages=[tf, idf])
    tfidf = pipeline.fit(DF)
    tfidfDF = tfidf.transform(DF)
    return tfidfDF



#Takes a DataFrame and returns a DataFrame with a column of Word Embedding Features.
def embeddingfeats(DF, inputCol='stemmed'):
    #Takes stemmed text DF and returns Word Embedding (word2vec) Features
    word2vec = Word2Vec(inputCol=inputCol, outputCol='features')
    embedding = word2vec.fit(DF)
    embeddingDF = embedding.transform(DF)
    return embeddingDF



#Takes a DataFrame and returns a DataFrame with a column of Properties-Based Features.
def propertiesfeats(DF):
    cat_cols = [item[0] for item in DF.dtypes if item[1]=='string']
    num_cols = [item[0] for item in DF.dtypes if 'int' in item[1]]
    bool_cols = [item[0] for item in DF.dtypes if item[1]=='boolean']
    #num_cols.remove('hasHTML')
    stages = []
    stringIndexer = [StringIndexer(inputCol = catcol, handleInvalid = 'keep', outputCol = catcol+'Index') for catcol in cat_cols]
    encoder = OneHotEncoderEstimator(inputCols=[ind.getOutputCol() for ind in stringIndexer],
                                                outputCols=[catcol+'classVec' for catcol in cat_cols])
    assemblerInputs = [c + 'classVec' for c in cat_cols] + num_cols + bool_cols
    if('label' in assemblerInputs): assemblerInputs.remove('label')
    if('id' in assemblerInputs): assemblerInputs.remove('id')
    assembler = VectorAssembler(inputCols = assemblerInputs, outputCol='features')
    #assembler = VectorAssembler(inputCols = assemblerInputs, outputCol='features')
    stages = stringIndexer + [encoder, assembler]
    pipeline = Pipeline(stages=stages)
    feats = pipeline.fit(DF)
    properties = feats.transform(DF)
    return properties



#Function to train and evaluate a combination of Feature Retrieving steps and a Classifier
def classtrain(DF, feat, classifier, classifiergrid=None, performance='nothing'):
    stime = time.time()
    result = MLModeler.pipemodeler(DF, feat, classifier, classifiergrid)
    result.train()
    if(performance=='nothing'):
        result.performance()
    else:
        result.performancerdd()
    print("Training took " + str(round(((time.time()-stime)/60), 2)) + " minutes to run.")
    return result



#This part of code is used so that the DataFrame is built only once, and if it is
#already built, its creation is skipped.
def trycreateDF(halflimit=2200):
    global DF
    try:
        print('Searching for Stemmed DF...')
        print(DF)
        print('Found it.')
    except:
        print('No Stemmed DF, creating one...')
        DF = createDF(halflimit)
    return DF

#DF = createDF(2200)

class Classifier:
    def __init__(self, classifier, classifiergrid):
        self.classifier = classifier
        self.classifiergrid = classifiergrid

def getclassifiers():
    #Classifiers used for training a model.
    lr = LogisticRegression(maxIter=10)
    lrgrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()
    LR = Classifier(lr, lrgrid)
    
    dt = DecisionTreeClassifier()
    dtgrid = ParamGridBuilder()\
    .addGrid(dt.maxDepth, [2, 5, 10])\
    .addGrid(dt.minInfoGain, [0.0, 0.1])\
    .addGrid(dt.maxBins, [6, 32])\
    .build()
    DT = Classifier(dt, dtgrid)
    
    rf = RandomForestClassifier()
    rfgrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [5, 20])\
    .addGrid(rf.maxDepth, [2, 5])\
    .build()
    RF = Classifier(rf, rfgrid)
    
    gbt = GBTClassifier(maxDepth=3, maxBins=16, maxIter=5)
    gbtgrid = ParamGridBuilder()\
    .addGrid(gbt.maxDepth, [2, 5])\
    .addGrid(gbt.maxIter, [5, 20])\
    .addGrid(gbt.stepSize, [0.01, 0.1])\
    .build()
    GBT = Classifier(gbt, gbtgrid)
    
    mpc = MultilayerPerceptronClassifier(layers=[6,4,2])
    mpcgrid = ParamGridBuilder()\
    .addGrid(mpc.maxIter, [20, 100])\
    .addGrid(mpc.tol, [1e-06, 1e-04])\
    .build()
    MPC = Classifier(mpc, mpcgrid)
    
    lsvc = LinearSVC(maxIter=15)
    lsvcgrid = ParamGridBuilder()\
    .addGrid(lsvc.threshold, [0.0, 0.05])\
    .build()
    LSVC = Classifier(lsvc, lsvcgrid)
    
    nb = NaiveBayes()
    nbgrid = ParamGridBuilder()\
    .addGrid(nb.smoothing, [0.2, 0.5, 1.0])\
    .build()
    NB = Classifier(nb, nbgrid)
    
    return [LR, DT, RF, GBT, LSVC, NB]
#    return [[lr, gridlr] , gbt, dt, rf, mpc, lsvc, nb]



#Creating the Properties Retriever Stages
def propstages(DF):
    DF = DF.drop('joinedLem')
    cat_cols = [item[0] for item in DF.dtypes if item[1]=='string']
    num_cols = [item[0] for item in DF.dtypes if ('int' in item[1] or 'float' in item[1] or 'double' in item[1])]
    bool_cols = [item[0] for item in DF.dtypes if item[1]=='boolean']
    
    #String Indexer takes the Categorical Columns (string types) and projects them to a number
    stringIndexer = [StringIndexer(inputCol = catcol, handleInvalid = 'keep', outputCol = catcol+'Index') for catcol in cat_cols]
    
    #OneHotEncoder maps a categorical feature, represented as a label index, to a binary vector
    #with at most a single one-value indicating the presence of a specific feature value from among
    #the set of all feature values.
    encoder = OneHotEncoderEstimator(inputCols=[ind.getOutputCol() for ind in stringIndexer],
                                                outputCols=[catcol+'classVec' for catcol in cat_cols])
    
    #VectorAssembler is a transformer that combines a given list of columns into a single vector column.
    #It is useful for combining multiple columns of features into one column.
    assemblerInputs = [c + 'classVec' for c in cat_cols] + num_cols + bool_cols
    if('label' in assemblerInputs): assemblerInputs.remove('label')
    if('id' in assemblerInputs): assemblerInputs.remove('id')
    assemblerInputs.remove('hasHTML')
    assembler = VectorAssembler(inputCols = assemblerInputs, outputCol='features')
    
    #Stages are used to Pipeline the transformations to the dataset with one command
    stages = stringIndexer + [encoder, assembler]
    return stages



#Create Text Based stages
def textstages(inputCol='stemmed'):
    #TF-IDF is a bag-of-words function, which gives higher weight to words that appear
    #frequently on a document but not frequently in all of the documents. The output is a features column.
    tf = HashingTF(inputCol=inputCol, outputCol='rawFeatures', numFeatures=500)
    idf = IDF(inputCol='rawFeatures', outputCol='features', minDocFreq=2.0)
    #idf = IDF(inputCol='rawFeatures', outputCol='features', minDocFreq=2.0)
    
    #Word2Vec is a Word Embedding function, which represents each word as a vector,
    #with words with similar meanings having neighboring vectors. The output is a feature column.
    word2vec = Word2Vec(inputCol=inputCol, outputCol='features')
    
    #Document Assembler to get Annotators (data type used by spark-NLP)
    docas = DocumentAssembler().setInputCol('joinedLem').setOutputCol('document')
    tok = Tokenizer().setInputCols(['document']).setOutputCol('token')
    #add BERT class
    bert = BertEmbeddings.pretrained('bert_base_cased', 'en').setInputCols(['document','token']).setOutputCol('bertFeatures')
    embfin = sparknlp.EmbeddingsFinisher()\
        .setInputCols('bertFeatures')\
        .setOutputCols('features')\
        .setOutputAsVector(True)
    embfinfin = EmbeddingsFinisherFinisher(inputCol='features', outputCol='features')
    return [[tf,idf], [word2vec], [docas, tok, bert, embfin, embfinfin]]





def appendselector(stages, percent=0.5):
    #A Chi-Square Feature Selector uses the Chi-Squared test of independence to decide which features
    #as the most "useful". In this case, 50% of the original amount of features are set to be kept.
    #With these Transformers, the stages for training Classifiers are set (different Transformer
    #for TF-IDF and Word Embedding Text-Based Features.
    if(percent<1.0):
        print("Appending Chi-Square to stages with percentage " + str(percent))
        selectorType = 'percentile'
        numTopFeatures = 50
        percentile = percent
    else:
        print("Appending Chi-Square to stage with numTopFeatures " + str(percent))
        selectorType = 'numTopFeatures'
        numTopFeatures = percent
        percentile = 0.1
        
    stages[-1].setOutputCol('prefeatures')
    selector = ChiSqSelector(numTopFeatures=numTopFeatures, featuresCol ='prefeatures', outputCol='features',
                             selectorType=selectorType, percentile=percentile)
    selectorstages = stages + [selector]
    return selectorstages



def traincombos(DF, feats, classifiers):
    for f in feats[:]:
        featlist = []
        for c in classifiers:
            print("NEXT ##################################")
            result = classtrain(DF, f, c.classifier, c.classifiergrid)
            print(result.model.bestModel.stages[-1].explainParams())
            result.printperformance()
            featlist.append(result)
        curvemetrics.plotCurves(featlist)



class Exploder(Transformer):
    def __init__(self, inputCol='features', outputCol='features'):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.uid = 'Exploder123'
    
    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn(self.outputCol, explode(self.inputCol))
        return df



# class EmbeddingsFinisherFinisher(Transformer, HasInputCol, HasOutputCol,
#                                  DefaultParamsReadable, DefaultParamsWritable):
#     @keyword_only
#     # def __init__(self, inputCol='features', outputCol='features'):
#     def __init__(self, inputCol=None, outputCol=None):
#         super(EmbeddingsFinisherFinisher, self).__init__()
#         # self._paramMap = {}
#         # self._params = {}
#         # self.inputCol = inputCol
#         # self.outputCol = outputCol
#         # self._setDefault(inputCol='features')
#         # self._setDefault(outputCol='features')
#         # self.uid = 'EmbeddingsFinisherFinisher_123'
#         kwargs = self._input_kwargs
#         self.setParams(**kwargs)
    
#     @keyword_only
#     # def setParams(self, inputCol='features', outputCol='features'):
#     def setParams(self, inputCol=None, outputCol=None):
#         kwargs = self._input_kwargs
#         return self._set(**kwargs)
    
#     def setInputCol(self, value):
#         return self._set(inputCol=value)
    
#     def setOutputCol(self, value):
#         return self._set(outputCol=value)
    
#     def _transform(self, df: DataFrame) -> DataFrame:
#     # def _transform(self, df):
#         # df = df.withColumn(self.outputCol, df[self.inputCol].getItem(0))
        
#         def None2EmptyVector(featCol):
#             # Getting 1st element because EmbeddingsFinisher outputs it like
#             #[<contents>]
#             nonlocal featSize
#             try:
#                 featCol = featCol[0]
#             except:
#                 featCol = SparseVector(featSize,{})
#             #Calculating Features Length
#             # featSize = 0
#             # i = 0
#             # while featSize == 0:
#             #     try:
#             #         featSize = len(df.select(featCol).collect()[i][0])
#             #     except:
#             #         featSize = 0
#             #     i += 1
#             # #Replacing empty cells with Features with values of 0
#             # if featCol == None:
#             #     featCol = DenseVector([0]*featSize)
#             return featCol
#         featSize = 0
#         i = 0
#         while featSize == 0:
#             try:
#                 # featSize = len(df.select(self.inputCol).collect()[i][0][0])
#                 featSize = len(df.select(self.getInputCol()).collect()[i][0][0])
#             except:
#                 featSize = 0
#             i += 1
#         udfNone2EmptyVector = udf(None2EmptyVector, VectorUDT())
#         # df = df.withColumn(self.outputCol, udfNone2EmptyVector(self.outputCol))
#         # df = df.withColumn(self.outputCol, udfNone2EmptyVector(self.inputCol))
#         df = df.withColumn(self.getOutputCol(), udfNone2EmptyVector(self.getOutputCol()))
#         return df
    


class EmbeddingsFinisherFinisher(Estimator, HasInputCol, HasOutputCol,
                                  DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(EmbeddingsFinisherFinisher, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    # def setInputCol(self, value):
    #     return self._set(inputCol=value)
    
    # def setOutputCol(self, value):
    #     return self._set(outputCol=value)
    
    def _fit(self, df):
        return EmbeddingsFinisherFinisherModel(inputCol=self.getInputCol(), outputCol=self.getOutputCol())

class EmbeddingsFinisherFinisherModel(Model, HasInputCol, HasOutputCol,
                                      DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(EmbeddingsFinisherFinisherModel, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _transform(self, df):
        
        def None2EmptyVector(featCol):
            # Getting 1st element because EmbeddingsFinisher outputs it like
            #[<contents>]
            nonlocal featSize
            try:
                featCol = featCol[0]
            except:
                featCol = SparseVector(featSize,{})
            return featCol
        
        featSize = 0
        i = 0
        while featSize == 0:
            try:
                featSize = len(df.select(self.getInputCol()).collect()[i][0][0])
            except:
                featSize = 0
            i += 1
        
        udfNone2EmptyVector = udf(None2EmptyVector, VectorUDT())
        df = df.withColumn(self.getOutputCol(), udfNone2EmptyVector(self.getInputCol()))
        return df
    
    



def getdata(halflimit=2200, inputCol="lemmatized"):
    global DF
    DF = trycreateDF(halflimit)
    #A list of Feature Retrievers and one of Classifiers is made to test every combination.
    feats = [propstages(DF)] + textstages(inputCol=inputCol)
    classifiers = getclassifiers()
    return DF, feats, classifiers



if __name__ == "__main__":
    DF, feats, classifiers = getdata(halflimit=20000, inputCol="lemmatized")
    traincombos(DF, feats, classifiers)


#start_time is set as time.time() at the start of the program in order to print the total time
#that was spent to run the entirety of the program.
print("Total Program took " + str(round(((time.time()-start_time)/60), 2)) + " minutes to run.")