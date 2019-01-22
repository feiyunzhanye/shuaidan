# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:18:01 2019
https://blog.csdn.net/laobai1015/article/details/80415080
@author: ChinaNet
"""
import sys
import os
import jieba
import cPickle as pickle
from sklearn.datasets import base

def savefile(savepath,content):
    with open(savepath,'wb') as fp:
        fp.write(content)

def readfile(path):
    with open(path,'rb') as fp:
        content = fp.read()
    return content

def corpus_segment(corpus_path,seg_path):
     ''''' 
    corpus_path是未分词语料库路径 
    seg_path是分词后语料库存储路径 
    '''  
    catelist = os.listdir(corpus_path)#获取corpus_path下的所有子目录
    
    for mydir in catelist:
        class_path = corpus_path +mydir +"/"
        seg_dir = seg_path +mydir +"/"
        
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        file_list = os.listdir(class_path)
      
        for file_path in file_list:
            fullname = class_path+file_path
            content = readfile(fullname)
            
            content = content.replace("\r\n","") #删除换行
            content = content.replace(" ","")
            content_seg = jieba.cut(content) #为文件内容分词
            savefile(seg_dir +file_path,"".join(content_seg))
    print("中文语料分词结束！！！")
def _readfile(path):
    with open(path,"rb") as fp:
        content = fp.read()
    return content
def corpus2Bunch(wordbag_path,seg_path):
    catelist = os.listdir(seg_path)
    bunch = base.Bunch(target_name=[],label=[],filename=[],content=[])
    bunch.target_name.extend(catelist)
    
    # 获取每个目录下所有的文件  
    for mydir in catelist:  
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径  
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件  
        for file_path in file_list:  # 遍历类别目录下文件  
            fullname = class_path + file_path  # 拼出文件名全路径  
            bunch.label.append(mydir)  
            bunch.filenames.append(fullname)  
            bunch.contents.append(_readfile(fullname))  # 读取文件内容  
            '''''append(element)是python list中的函数，意思是向原来的list中添加element，注意与extend()函数的区别'''  
    # 将bunch存储到wordbag_path路径中  
    with open(wordbag_path, "wb") as file_obj:  
        pickle.dump(bunch, file_obj)  
    print "构建文本对象结束！！！"  

wordbag_path = "data/train_word_bag/train_set.dat"  # Bunch存储路径  
seg_path = "data/train_corpus_seg/"  # 分词后分类语料库路径  
corpus2Bunch(wordbag_path, seg_path)  
  
    # 对测试集进行Bunch化操作：  
wordbag_path = "data/test/test_word_bag/test_set.dat"  # Bunch存储路径  
seg_path = "data/test/test_corpus_seg/"  # 分词后分类语料库路径  
corpus2Bunch(wordbag_path, seg_path) 

if __name__ == "_main_"

    corpus_path = "data/train/"
    seg_path = "data/train_corpus_seg/"
    corpus_segment(corpus_path,seg_path)
    
    