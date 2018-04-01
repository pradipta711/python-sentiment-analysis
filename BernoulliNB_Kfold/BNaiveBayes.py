# -- coding: utf-8 --
"""
Created on Tue Mar 27 02:11:30 2018

@author: Pradipta
"""

#  Reading a text-based dataset into pandas
import nltk
import pandas as pd


# read file into pandas from the working directory
dataset = pd.read_csv('full-corpusf.csv', header=None, names=['Topic','Sentiment','TweetId','Date','Tweet'])
dataset['label_num'] = dataset.Tweet.map({'positive':0,'negative':1, 'neutral':3, 'irrelevant':4})

'''
#Cleaning all tweets
corpus = []
for i in range(0,4134):
    tweet = re.sub('[^a-zA-Z]]','', dataset['TweetText'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps= PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

'''

X = dataset.Tweet
y = dataset.Sentiment
print(X.shape)
print(y.shape)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
vect =CountVectorizer(ngram_range=(1,2),
                             stop_words='english')
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm

X_test_dtm = vect.transform(X_test)
X_test_dtm


from sklearn.feature_extraction.text import TfidfTransformer

tfidf  = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_dtm)
X_test_tfidf =tfidf.transform(X_test_dtm) 


#from sklearn.linear_model import SGDClassifier
#sdg = SGDClassifier()

from sklearn.naive_bayes import BernoulliNB
BernoulliNBclassifier = BernoulliNB(alpha = 0.01)

# train the model using X_train_dtm
BernoulliNBclassifier.fit(X_train_tfidf, y_train)


# make class predictions for X_test_dtm
y_pred_class =BernoulliNBclassifier.predict(X_test_tfidf)


# calculate accuracy,recall and precision of class predictions
from sklearn import metrics
acc =metrics.accuracy_score(y_test, y_pred_class)
recall=metrics.recall_score(y_test,y_pred_class,average='macro')
precision=metrics.precision_score(y_test,y_pred_class,average='macro')
 
print(acc)
print(recall)
print(precision)

# print the confusion matrix
cm4 = metrics.confusion_matrix(y_test, y_pred_class)


