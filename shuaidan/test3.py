# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:56:56 2019

@author: ChinaNet
"""

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import  fetch_20newsgroups

'''
1 读取数据部分
'''
# 该api会即使联网下载数据
news = fetch_20newsgroups(subset="all")
# 检查数据规模和细节
# print(len(news.data))
# print(news.data[0])
'''
18846
From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>
Subject: Pens fans reactions
Organization: Post Office, Carnegie Mellon, Pittsburgh, PA
Lines: 12
NNTP-Posting-Host: po4.andrew.cmu.edu
I am sure some bashers of Pens fans are pretty confused about the lack
of any kind of posts about the recent Pens massacre of the Devils. Actually,
I am  bit puzzled too and a bit relieved. However, I am going to put an end
to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
are killing those Devils worse than I thought. Jagr just showed you why
he is much better than his regular season stats. He is also a lot
fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
regular season game.          PENS RULE!!!
'''

'''
2 分割数据部分
'''
x_train, x_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.25,
                                                    random_state=33)
#提取特征向量
tfid_vec = TfidfVectorizer()
x_tfid_train = tfid_vec.fit_transform(x_train)
x_tfid_test = tfid_vec.transform(x_test)

'''
3 贝叶斯分类器对新闻进行预测
'''
# 初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 训练集合上进行训练， 估计参数
mnb.fit(x_tfid_train, y_train)
# 对测试集合进行预测 保存预测结果
y_predict = mnb.predict(x_tfid_test)

'''
4 模型评估
'''
print("准确率:", mnb.score(x_test, y_test))
print("其他指标：\n",classification_report(y_test, y_predict, target_names=news.target_names))