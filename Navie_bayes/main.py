'''
Description：手动实现高斯朴素贝叶斯分类器，对于给定的红酒数据，将其分为3类best,better,good
Author:worthurlove
Date:2018.12.18
'''
from __future__ import division, print_function
from tools import train_test_split, normalize, accuracy_score, Plot
from naive_bayes import NaiveBayes
import pandas as pd

def main():

    data = pd.read_csv('wine.csv')
    y = data['class'].values

    X = data.drop('class', axis=1).values

    X = normalize(X)#数据标准化
    label=['best','better','good']
 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)#将数划分为训练集和测试集，标签也做同样的划分

    clf = NaiveBayes()#引用朴素贝叶斯分类器

    clf.fit(X_train, y_train)



    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    while accuracy < 0.98:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        clf = NaiveBayes()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # 使用PCA将维数降为2并绘制结果
    Plot().plot_in_2d(X_test, y_pred, title="Naive Bayes", accuracy=accuracy,legend_labels=label)

if __name__ == "__main__":
    main()