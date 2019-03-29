# -- coding: utf-8 --
"""
Created on Tue Mar 27 02:11:30 2018

@author: Pradipta
"""

import nltk
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# read file into pandas from the working directory
dataset = pd.read_csv('full-corpusf.csv', header=None, names=['Topic','Sentiment','TweetId','Date','Tweet'])
dataset['label_num'] = dataset.Tweet.map({'positive':0,'negative':1, 'neutral':2, 'irrelevant':3})

#Cleaning all tweets
corpus = []
for i in range(0,4134):
    tweet = re.sub('[^a-zA-Z]',' ', dataset['Tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps= PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

from sklearn.feature_extraction.text import CountVectorizer
vect =CountVectorizer(max_features=1500)
X = vect.fit_transform(corpus).toarray()
y= dataset.Sentiment


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, random_state=1)


#from sklearn.linear_model import SGDClassifier
#sdg = SGDClassifier()

from sklearn.naive_bayes import BernoulliNB
BernoulliNBclassifier = BernoulliNB(alpha = 0.01)

# train the model using X_train
BernoulliNBclassifier.fit(X_train, y_train)


# make class predictions for X_test
y_pred_class =BernoulliNBclassifier.predict(X_test)
print(y_pred_class)
print(y_test)

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

