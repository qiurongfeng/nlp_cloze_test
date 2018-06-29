# -*- coding: utf-8 -*-
"""
Created on 2018/5/24 21:01
author: Rongfeng.Qiu
file:main.py
"""
import data_preprocess
import train_model
import test_data_score


##数据预处理参数##
#load train_data
train_data = 'train_data'
#save path
midfile_savepath = 'middle_data/middle_file.txt'

##训练模型参数##
#model save path
model_savepath = 'model/model_1'
#特征向量的维度
size = 700
#词频少于min_count次数的单词会被丢弃掉
min_count = 2
#如果为1则会采用hierarchica·softmax技巧
hs = 1
#用于设置多少个noise words
negative = 0

#测试精度路径设置 
test_data_que = 'test_data/development_set.txt'
test_data_ans = 'test_data/development_set_answers.txt'

#需要输出答案的测试集和答案保存路径
test_set = 'test_data/test_set.txt'
save_ans_path = 'test_data/test_set_ans_xxxooss.txt'
#数据预处理
data_preprocess.file_text_preprocess(train_data,savepath=midfile_savepath)

#开始训练
train_model.train_model(midfile_savepath,model_savepath,size=size,min_count=min_count,hs=hs,negative=negative)

#测试精度
test_data_score.output_accuracy(model_savepath,test_data_que,test_data_ans)

#输出答案文件,outputfile = True 表示要输出文件
test_data_score.output_pro_ans(model_savepath,test_set,output_file=True,savepath=save_ans_path)