# -*- coding: utf-8 -*-
"""
业务需求：对工单文本内容进行分析判断，判断工单类型
@author: feiyun
"""

import pandas as pd
import time
import jieba
import jieba.analyse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# 分类报告：precision/recall/fi-score/均值/分类个数
from sklearn.metrics import classification_report
file_data_path = "D:/12345/12345_all.csv"
#加载停用词
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

#实现中文分词
def wordsSplit(text):
    stopwords = stopwordslist('data/stop_words_ch.txt')
    datas = []
    data_no_stop = []
    jieba_words = jieba.lcut(text)
    #jieba_words = jieba.analyse.extract_tags(text, topK=20, withWeight=False)
    #print('text:', jieba_words)
    outstr = ''
    outarray= []#按数组输出
    for word in jieba_words:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
                outarray.append(word)
    datas.append(outstr)
    data_no_stop.append(' '.join(jieba_words))
    return ' '.join(datas)
    #return outarray

f = open(file_data_path,'r',encoding='utf-8')
df = pd.read_csv(f,encoding='utf-8')  # 读入数据
data_content = df.iloc[:, 24:25].values.astype(str)
data_type = df.iloc[:, 19:20].values.astype(str)
data_id = df.iloc[:, 12:13].values.astype(str)#读入id
type_id = {'其他类':0,'求助类':1,'投诉举报类':2,'意见建议类':3,'咨询类':4}
#工单类型文字转数字
for i in range(0,len(data_type)):
    if data_type[i][0] in type_id.keys():
        data_type[i] = type_id[data_type[i][0]]
    else:
        data_type[i] = type_id['其他类']
features = []
results = []
for type in data_type:
    results.append(type[0])
tfidf_words = ""
for content in data_content:
    words = wordsSplit(content[0])
    features.append(words)
    tfidf_words += words
    tfidf_words += " "

# 对结果集 特征集随机抽取 分为测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(features, results, test_size=0.2,random_state=0)

#转换文本数据
tdf = TfidfVectorizer(tfidf_words)
X1 = tdf.fit_transform(X_train)
clf = RandomForestClassifier()
clf.fit(X1, y_train)

start = time.time()
X_test_tdf = tdf.transform(X_test)
y_pred =clf.predict(X_test_tdf)
test_report = classification_report(y_test,y_pred)
print("打印报告")
print(test_report)
end = time.time()
print(end-start)
