# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:50:56 2019

@author: ChinaNet
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers

data = open('data/reviews_data').read()

labels, texts = [], []

for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])    
    texts.append(content[1])



#创建一个dataframe，列名为text和label

trainDF = pandas.DataFrame()

trainDF['text'] = texts

trainDF['label'] = labels

#将数据集分为训练集和验证集

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])



# label编码为目标变量

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)

#创建一个向量计数器对象

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(trainDF['text'])

#使用向量计数器对象转换训练集和验证集
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
# ngram 级tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



#词性级tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['text'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)