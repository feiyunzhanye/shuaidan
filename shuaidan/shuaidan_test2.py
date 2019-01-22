# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:04:53 2019

@author: ChinaNet
"""

import pandas as pd
import jieba
import jieba.analyse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 模型评估模块
from sklearn.metrics import classification_report
jieba.analyse.set_stop_words("data/stop_words.txt")
def getWords(contents):
    words_all = []
    for content in contents:
        words = jieba.analyse.extract_tags(content,topK=30, withWeight=False, allowPOS=('n','v'))
        words_all.append(words)
    document = [" ".join(sent0) for sent0 in words_all]
    return document

file_model_path = "data/test1.csv"
file_pred_path = "data/test11.csv"
file_data = "data/shuaidan2.csv"
#columns = ['甩单流水号','甩单业务类型id','甩单业务类型','甩单备注','业务类型','订单1']
#data_model = pd.read_csv(file_model_path,names=columns,encoding='GBK')
#data_test = pd.read_csv(file_pred_path,names=columns,encoding='GBK')
#data_model.drop('fnlgwt', axis=1, inplace=True)
data = pd.read_csv(file_data,encoding='gb18030')

X = data['甩单备注'].astype(str)
#types = data_model['甩单业务类型']
y = data['订单数'].astype(str)
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)
X_tdf = getWords(X_train)
"""
TfidfVectorizer参数
当设置为浮点数时，过滤出现在超过max_df/低于min_df比例的句子中的词语
"""
onehot = TfidfVectorizer()#token_pattern=r"(?u)\b\w\w+\b"其中的两个\w决定了其匹配长度至少为2的单词
X1 = onehot.fit_transform(X_tdf)

# 初始化朴素贝叶斯模型
#mnb = MultinomialNB()
#mnb.fit(X1,y)
clf = RandomForestClassifier()
clf.fit(X1,y_train)
print("模型完成")


#X_importance = clf.feature_log_prob_

#X_pred = data_test['甩单备注']
#types_pred = data_test['甩单业务类型']
X_pred_tdf = getWords(X_test)
X_pred2 = onehot.transform(X_pred_tdf)
y_pred = clf.predict(X_pred2)
#print(RandomForestClassifier.score(X_pred2, y_test))
print(classification_report(y_test,y_pred))

