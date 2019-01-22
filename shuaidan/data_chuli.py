# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 01:19:56 2019

@author: feiyun
"""

import csv
import pandas as pd
path_shuai = "data/test1.csv"
path_ding = "data/dingdan2.csv"
path_out = "data/out2.csv"
reader_shuaidan = []
reader_order = []
flag = []
#reader_shuaidan = pd.read_csv(path_shuai,encoding='gb18030')
#reader_order = pd.read_csv(path_ding,encoding='gb18030')
#print(reader_shuaidan)
#print(reader_order)
with open(path_shuai,'r',encoding='gb18030') as f:
    reader_shuaidan1 = csv.reader(f)
    for row in reader_shuaidan1:
        reader_shuaidan.append(row)
with open(path_ding,'r',encoding='gb18030') as f2:
    reader_order1 = csv.reader(f2) 
    for row in reader_order1:
        reader_order.append(row)   
print("开始读写")       
for row1 in reader_shuaidan:
    data1 = row1[0]
    flag_one = []
    for row2 in reader_order:
        if(data1==row2[10]):
            flag_one.append(row2[7])
    flag.append(flag_one)
print("读取完成")   
with open(path_out,'w',newline='',encoding='gb18030') as f3:
    writer = csv.writer(f3)
    for row in flag:
        writer.writerow(row)
print("写入完成")