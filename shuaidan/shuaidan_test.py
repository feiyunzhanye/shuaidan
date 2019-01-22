# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:25:35 2019

@author: ChinaNet
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
df = pd.DataFrame({'name':['Lily','Lucy','Jim','Tom','Anna','Jack'],'weight':[42,38,78,67,52,80],'height':[162,158,169,170,165,175],'is_fat':[0,0,1,0,1,0]})
print(df)

X = df.loc[:,['weight','height']]
y = df['is_fat']
print(X)

clf = RandomForestClassifier();
clf.fit(X,y)

X_importance = clf.feature_importances_
print(X_importance)

df_pred = pd.DataFrame({'name':['Anna','Jack','Sam'],'weight':[52,80,92],'height':[165,175,178],'is_fat':[1,0,1]})
X_pred = df_pred.loc[:,['weight','height']]
y_pred = clf.predict(X_pred)
print(y_pred)

plt.figure()
df_pred['is_fat_pred'] = y_pred
df_0 = df_pred[df_pred['is_fat_pred']==0]
df_1 = df_pred[df_pred['is_fat_pred']==1]
plt.scatter(df_0['weight'],df_0['height'],c='y',s=50,label='normal')
plt.scatter(df_1['weight'],df_1['height'],c='lightblue',s=100,label='fat')
for k in range(len(X_pred)):
    plt.text(X_pred['weight'][k],X_pred['height'][k],df_pred['name'][k])
    
plt.legend()