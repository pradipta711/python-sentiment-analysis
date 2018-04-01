# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 02:11:30 2018

@author: Pradipta
"""

# ## Part 3: Reading a text-based dataset into pandas
import nltk
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import nltk
import re
#nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# read file into pandas from the working directory
dataset = pd.read_csv('full-corpusf.csv', header=None, names=['Topic','Sentiment','TweetId','Date','Tweet'])

# convert label to a numerical variable
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
#print(X.shape)
#print(np.unique(y))

from sklearn.feature_extraction.text import CountVectorizer
vect =CountVectorizer(ngram_range=(1,2),
                             stop_words='english')
X_dtm = vect.fit_transform(X)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf  = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_dtm)

#from sklearn.naive_bayes import GaussianNB
#gaus = GaussianNB()

from sklearn.naive_bayes import BernoulliNB
BernoulliNBclassifier = BernoulliNB(alpha = 0.01)


#from sklearn.linear_model import SGDClassifier
#svm = SGDClassifier()

list_acc=[]
list_recall=[]
list_precision=[]
kf=KFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(X_tfidf):
    
    X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # train the model using X_train_dtm
    BernoulliNBclassifier.fit(X_train, y_train)

    # make class predictions for X_test_dtm
    y_pred_class = BernoulliNBclassifier.predict(X_test)
    
    # calculate accuracy of class predictions

    acc =metrics.accuracy_score(y_test, y_pred_class)
    list_acc.append(acc)
    recall=metrics.recall_score(y_test,y_pred_class,average='macro')
    list_recall.append(recall)
    precision=metrics.precision_score(y_test,y_pred_class,average='macro')
    list_precision.append(precision)
print(np.average(list_acc))
print(np.average(list_recall))
print(np.average(list_precision))

N=3    
metrics_values_kfold=[list_acc[0],list_recall[0],list_precision[0]]
metrics_values=[0.75,0.66,0.66]
ind = np.arange(N)  # the x locations for the groups
width = 0.25   

fig, ax = plt.subplots()
rects1 = ax.bar(ind, metrics_values_kfold, width, color='y')
rects2 = ax.bar(ind+width, metrics_values, width, color='b')

ax.legend((rects1[0], rects2[0]), ('Metrics(KFold)', 'Metrics(Without KFold)'))

ax.set_title('BernoulliNBclassifier')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Accuracy', 'Recall', 'Precision'))
plt.show()




