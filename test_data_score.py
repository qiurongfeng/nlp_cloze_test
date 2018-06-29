# -*- coding: utf-8 -*-
"""
Created on 2018/5/24 20:48
author: Rongfeng.Qiu
file:test_data_score.py
"""

from gensim.models import Word2Vec
import argparse
import gensim
import re

# outputfile = ,True 表示要输出文件,False 表示只要做精度测试，生成列表即可
def output_pro_ans(model_name,test_date_que,output_file = False,savepath = ''):
    print('predict ans...')
    #load model  
    model = Word2Vec.load(model_name)
    #load test_data
    res = []
    res_file = []
    with open(test_date_que,'r') as f:
        content = f.readlines()
        #去除空行
        content_list = ([l.strip() for l in content if l.strip() != ''])
        line_num = len(content_list)
        #每六行一道题
        for index in range(0,line_num,6):
            ans_sen = content_list[index]
            ans = []
            check_list = []
            for i in range(index + 1,index + 6):
                ans.append(content_list[i])
            for word in ans:
                #将词带入句子
                check_sen = ans_sen.replace('_____',word.split()[1]).split()
                check_list.append(check_sen)
            #对句子进行打分并输出得分最高的词
            raw_res = model.score(check_list).tolist()
            index = raw_res.index(max(raw_res))
            res.append(ans[index].split()[1])
#if output_file is true.输出文件
            if output_file:
                res_file.append(ans[index])
    if output_file:
        try:
            with open(savepath,'w') as f:
                write_line = []
                i = 1
                for line in res_file:
                    line = line.replace(')','')
                    write_line.append(str(i) + " " + str(line) + '\n')
                    i += 1
                f.writelines(write_line)
            print("success! output file is already down.")
        except:
            print("error. please input save path...")
########################
    return res

def output_accuracy(model_name,test_date_que,test_data_ans):
    res_pro_ans = output_pro_ans(model_name,test_date_que)
    res_ans = []
    #提取正确答案列表
    with open(test_data_ans,'r') as fans:
        ans_list = fans.readlines()
        for line in ans_list:
            res_ans.append(line.strip().split()[2])

    #列表对比进行精度测试
    ac_num = 0
    num = len(res_ans)
    for index in range(num):
        if res_pro_ans[index] == res_ans[index]:
            ac_num += 1
    print('ac number:' + str(ac_num))
    print('all num:' + str(num))
    print("accuracy:" + str(float(ac_num)/num))


if __name__ == '__main__':
    model_name = '/home/qrf/Desktop/NLP_PRO_PRO/model_700_2_0.6/model_finally_2'
    test_data_que = 'test_data/development_set.txt'
    test_data_ans = 'test_data/development_set_answers.txt'
    #output_accuracy(model_name,test_data_que,test_data_ans)
    output_pro_ans('/home/qrf/Desktop/NLP_PRO_PRO/model_700_2_0.6/model_finally_2','/home/qrf/Desktop/nlp_release/test_data/test_set.txt',output_file=True,savepath = 'test_set_ans.txt')
