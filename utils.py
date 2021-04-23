import re
import mailbox
from bs4 import BeautifulSoup
import snowballstemmer
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml import Pipeline

import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet



#Get the email and return a dictionary list with "mimeType" and "payload"
#of every part of the email.
def getpayload_dict(msg):
    return __getpayload_dict_rec__(msg, [])


def __getpayload_dict_rec__(msg, payloadresult):
    payload = msg.get_payload()
    
    if(msg.get('content-transfer-encoding') == 'base64'):
        payload = msg.get_payload(decode=True)
    
    if msg.is_multipart():
        for subMsg in payload:
            __getpayload_dict_rec__(subMsg, payloadresult)
    else:
        payloadresult.append({"mimeType": msg.get_content_type(), "payload": payload})
    return payloadresult

 

#Prints the content types of every part of the email.
def print_content_types(msg):
    if(msg.is_multipart()):
        print("\nMultipart:")
        for submsg in msg.get_payload():
            print(submsg.get_content_type())
    else:
        print("\nSingle:")
        print(msg.get_content_type())



#Takes a message and transforms any HTML part to plain text.
def unhtmlify(msg): 
    payload_dict = getpayload_dict(msg)
    for part in payload_dict:
        # if(part['mimeType'] in ['text/html','text/plain']):
        if('text' in part['mimeType'] or 'multipart' in part['mimeType']):
            soup = BeautifulSoup(part['payload'], "lxml")
            #Getting just the text didn't clear scripts and styles,
            #so we clear them manually.
            for s in soup(["script","style"]):
                s.extract()
            clean_text = ' '.join(soup.stripped_strings)
            part['payload'] = clean_text
    return payload_dict



#Returns True if the message entered is empty
def is_empty(msg):
    payload_dict = getpayload_dict(msg)
    totalsize = 0
    for part in payload_dict:
        totalsize += len(re.sub(r'\s+','',str(part['payload'])))
    if(totalsize < 1): return True
    else: return False



#Returns every text part of the email as a string
def get_email_text(msg):
    unhtmlified = unhtmlify(msg)
    email_text = ""
    for part in unhtmlified:
        #Append only the text parts of the email.
        # if(part['mimeType'] in ['text/plain','text/html']):
        if('text' in part['mimeType'] or 'multipart' in part['mimeType']):
            email_text += part['payload']
    return email_text



#Clean the text of the email
def clean_text(text):
    #Some emails had an "=\n" every certain amount of characters
    cleantext = text.replace('=\n','')
    #clean URLs
    urlregex = r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    urlpattern = re.compile(urlregex)
    cleantext = re.sub(urlpattern, 'url', cleantext)
    return cleantext



#Returns the text parts of the email, cleaned.
def get_clean_text(msg):
    email_text = get_email_text(msg)
    email_text = clean_text(email_text)
    return email_text



#Used as a UDF (User Defined Function) for the textDF2setDF function
def stem2(in_vec):
    stemmer = snowballstemmer.stemmer('english')
    out_vec = []
    for x in in_vec:
        to_out = stemmer.stemWord(x)
        if(len(to_out) > 2):
            out_vec.append(to_out)
    return out_vec



#Used as a UDF (User Defined Function) for the textDF2setDF function
def lem2(in_vec):
    
    def pos_tagger(nltk_tag): 
        if nltk_tag.startswith('J'): 
            return wordnet.ADJ 
        elif nltk_tag.startswith('V'): 
            return wordnet.VERB 
        elif nltk_tag.startswith('N'): 
            return wordnet.NOUN 
        elif nltk_tag.startswith('R'): 
            return wordnet.ADV 
        else:           
            return None
    
    lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(in_vec)
    
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged)) 
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if len(word) > 2:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag)) 
    return lemmatized_sentence



#Takes the DataFrame which has strings of cleaned text, keeps only the words, removes the
#words that don't provide information (I, you, the, in etc.) and stems the rest of them
#(e.g. "running" becomes "run").
def textDF2setDF(textDF, textCol):
    #gaps=False means that the tokenizer grabs whichever substring MATCHES the pattern.
    #Otherwise, the tokenizer would grab whichever substring is between matches of the pattern.
    regexTokenizer = RegexTokenizer(inputCol=textCol, outputCol="words", pattern="[a-zA-Z]+", gaps=False)
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="stopWremoved")
    stages = [regexTokenizer, stopwordsRemover]
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(textDF)
    textDF = pipelineModel.transform(textDF)
    
    lemmatizer_udf = udf(lem2, ArrayType(StringType()))
    textDF = textDF.withColumn("lemmatized", lemmatizer_udf("stopWremoved"))
    
    stemmer_udf = udf(stem2, ArrayType(StringType()))
    textDF = textDF.withColumn("stemmed", stemmer_udf("stopWremoved"))
    
    join_udf = udf(lambda x: ",".join(x))
    textDF = textDF.withColumn("joinedLem", join_udf(col("lemmatized")))
    return textDF