# -*- coding: utf-8 -*-
"""
Created on 2018/5/24 14:24
author: Rongfeng.Qiu
file:train_model.py
"""

import nltk
from gensim.models import Word2Vec

def train_model(middle_file,savepath, size=100, alpha=0.025, window=5, min_count=5, sample=1e-3,min_alpha=0.0001,sg=0,hs=0,negative=5,cbow_mean=1,iter=5):
    print(" train start...")
    content_list = []
    word_list = []
    #打开预处理的文件
    with open(middle_file,'r') as f:
        content_list = f.readlines()
    #分词
    for sent in content_list:
        word_list.append(nltk.word_tokenize(sent))
    print("word token ready next to train...")
    #训练词向量,保存模型
    model_test = Word2Vec(size=size, alpha=alpha, window=window, min_count=min_count,  sample=sample,  min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, iter=iter)
    model_test.build_vocab(word_list)
    model_test.train(word_list,total_examples=model_test.corpus_count, epochs=model_test.iter)
    model_test.save(savepath) 
    print("success! The model name is " + savepath.split('/')[-1]) 

if __name__ == '__main__':
    train_model('middle_data/combine_sent_2.txt','model/model_1',size=700,min_count=2,hs=1,negative=0)
    
