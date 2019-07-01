'''
Description:对术语在线网站（http://www.termonline.cn）中查询结果的分页表格数据进行爬取
Author:worthurlove
Date:2019.7.1
'''
import time
import re
import pandas as pd
from urllib import request


#去除数据里的<span></span>标识符
def deleteSpan(s):
    clean_s = ''
    i = 0
    while i < len(s):
        if s[i] == '<':
            i += 1
            while s[i] != '>':
                i += 1
            i += 1
        else:
            clean_s += s[i]
            i += 1
    return clean_s


#数据处理第一步，去掉空白行
def deleteSpace(htmlSplit):
    step_1 = []
    for i in htmlSplit:
        if i != '':
            step_1.append(i)
    return step_1

#数据处理第二步，去除别名
def deleteAlsoName(step_1):
    step_2 = []
    i = 0
    while i < len(step_1):
        if step_1[i] == 'also_name':
            i += 1
            while step_1[i] != '[]' and step_1[i] != ']':
                i += 1
            i += 1
        step_2.append(step_1[i])
        i += 1
    return step_2


#数据处理第二步，根据CN和EN字段找出对应的数据
def getCnEn(step_2):
    step_3 = []
    for i in range(len(step_2)):
        if step_2[i] == 'cn':
            step_3.append(deleteSpan(step_2[i + 1]))
        if step_2[i] == 'en':
            if step_2[i + 1] != '[':
                step_3.append(deleteSpan(step_2[i + 1]))
    return step_3[:-10]

#中英文数据分离
def CnEnSplit(step_3):
    cn = []
    en = []
    for i,j in zip(step_3[0::2],step_3[1::2]):
        cn.append(i)
        en.append(j)
    return cn,en

#获取分页表格数据的总页数
def getPageNumber(alpha = 'A'):
    url = 'http://www.termonline.cn/list.jhtm?op=query&k=A&start=0&pageSize=15&sort=&resultType=0&conds%5B0%5D.key=all&conds%5B0%5D.match=1&conds%5B1%5D.val=&conds%5B1%5D.key=category&conds%5B1%5D.match=1&conds%5B2%5D.val=&conds%5B2%5D.key=subject_code&conds%5B2%5D.match=3&conds%5B3%5D.val=&conds%5B3%5D.key=publish_year&conds%5B3%5D.match=1&conds%5B0%5D.val='+alpha
    response = request.urlopen(url,timeout=30)
    html = response.read()
    html = html.decode('utf-8')
    #对数据进行切割
    htmlSplit = re.split('[{,}:"]',html.strip())
    return int(htmlSplit[-2])   


def getInfo(start = 0,alpha = 'A'):
    url = 'http://www.termonline.cn/list.jhtm?op=query&k=A&start='+str(start)+'&pageSize=15&sort=&resultType=0&conds%5B0%5D.key=all&conds%5B0%5D.match=1&conds%5B1%5D.val=&conds%5B1%5D.key=category&conds%5B1%5D.match=1&conds%5B2%5D.val=&conds%5B2%5D.key=subject_code&conds%5B2%5D.match=3&conds%5B3%5D.val=&conds%5B3%5D.key=publish_year&conds%5B3%5D.match=1&conds%5B0%5D.val='+alpha
    response = request.urlopen(url,timeout=30)
    html = response.read()
    html = html.decode('utf-8')
    #对数据进行切割
    htmlSplit = re.split('[{,}:"]',html.strip())

    step_1 = deleteSpace(htmlSplit)

    step_2 = deleteAlsoName(step_1)

    step_3 = getCnEn(step_2)

    return CnEnSplit(step_3)

if __name__ == '__main__':
    #遍历26个字母
    alphaNumber = 65
    while alphaNumber < 90:
        cn = []
        en = []
        start = 0
        alpha = chr(alphaNumber)
        #遍历每一页
        pageNumber = getPageNumber(alpha)

        i = 0
        while i < pageNumber:
            start = 15*i
            a,b = getInfo(start,alpha)
            cn.extend(a)
            en.extend(b)
            #降低访问频率，避免被反爬虫机制禁止
            time.sleep(1)
            i += 1
        
        data = {'cn':cn,'en':en}

        df = pd.DataFrame(data)

        print(df)

        df.to_excel('E:/shuyu_'+alpha+'.xlsx')

        alphaNumber += 1




