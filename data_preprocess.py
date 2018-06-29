# -*- coding: utf-8 -*-
"""
Created on 2018/5/24 10:27
author: Rongfeng.Qiu
file:data_preprocess.py
"""

import re
from nltk.stem import PorterStemmer
import os
import nltk
import gc

##标点
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

##停词
stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this',
              'that','these','those','then','just','so','than','such','both','through',
              'about','for','is','while','during','to','What','Which','Is','If','While','This']

#对文本的简单替换处理和去停词标点和提词干
def text_dispose(text,stop_words_remove = True,stem_words = False):
    ##文本的简单替换处理
    # Clean the text
    text = text.lower()
    text = re.sub("\n"," ",text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    text = re.sub('\d','',text)

    ##去标点
    text = ''.join([c for c in text if c not in punctuation])

    ##去停词
    if stop_words_remove:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    ##提词干
    if stem_words:
        text_list = text.split()
        stemmer = PorterStemmer()
        stem_words = [stemmer.stem(word) for word in text]
        text = " ".join(stem_words)
    return(text)

##训练集目录
def file_text_preprocess(dirpath = "train_data",savepath="middle_data/combine_sent.txt",stop_words_remove = True,stem_words = False):
    print('preprocess start ...')
    #列出文件夹文件
    files = os.listdir(dirpath)
    print(dirpath)
    i = 1 
    content_list = []
    #循环读取文件
    for filename in files:
        print( i , filename)
        i += 1
        if not os.path.isdir(filename):
            try:
                with open(dirpath + '/' + filename,'r',encoding = 'utf-8') as f:
                    temp = f.read()
            except:
                try:
                    with open(dirpath + '/' + filename,'r',encoding = 'GBK') as f:
                        temp = f.read()
                except:
                    with open(dirpath + '/' + filename,'r',encoding = 'ISO-8859-1') as f:
                        temp = f.read()
        #分句
        sen_list = nltk.sent_tokenize(str(temp))
        temp_wordlsit = []
        #分句后对每个句子预处理
        for sent in sen_list:
            temp_wordlsit.append(text_dispose(sent,stop_words_remove=stop_words_remove,stem_words=stem_words))
            #print(temp_wordlsit)
        content_list.extend(temp_wordlsit)
        print("success")
        del temp_wordlsit
        del sen_list
        gc.collect()
    print("data ready")
    #保存预处理完后的文件
    with open(savepath,'w') as f:
        writer_list = [] 
        for line in content_list:
            writer_list.append(line + '\n')
        f.writelines(writer_list)
    print("create middle_file success!")

if __name__ == '__main__':
    file_text_preprocess("train_data","middle_data/combine_sent_2.txt")
