# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:39:31 2019

@author: ChinaNet
"""
import csv


#待处理文件名字
filename_shuaidan = "data/甩单数据-训练数据.csv" 
filename_order = "data/订单数据-训练数据.csv"
filename_ceshi = "data/ceshi.csv"
filename_test = "data/test.csv"
data_shuaidan = []
    
with open(filename_shuaidan,'r',encoding='gb18030') as f:
    reader_shuaidan = csv.reader(f)
    data_shuaidan_id_remark= []
    print("开始读取")
    for row in reader_shuaidan:
        data_shuaidan_id_remark= []
        data_shuaidan_id_remark.append(row[0])
        data_shuaidan_id_remark.append(row[20])
        data_shuaidan.append(data_shuaidan_id_remark)
    print("读取完成")
print("开始写入:")
with open(filename_ceshi,'w',newline='',encoding='gb18030') as f3:
    writer = csv.writer(f3)
    for row in data_shuaidan:
        writer.writerow(row)
    print("写入完成")
with open(filename_order) as f2:
    reader_order = csv.reader(f2)
    data_order = []
    for row1 in reader_order:
        data_order.append(row1[0])
    

