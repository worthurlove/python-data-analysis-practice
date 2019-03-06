from __future__ import division, print_function
import numpy as np
import math
from decimal import *
getcontext().prec = 50

class NaiveBayes():
    """高斯朴素贝叶斯分类器. """
    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y) #3
        self.parameters = []
        # 计算每个类的每个要素的均值和方差
        for i, c in enumerate(self.classes):
            # 仅选择标签等于给定类的行
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            # 为每个要素添加均值和方差（列）
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}#平均值和方差

                self.parameters[i].append(parameters)

    def calculate_likelihood(self, mean, var, x):
        """ 给定均值和变量的数据x的高斯分布，算其概率值"""
        # print('均值：{}方差：{}数据：{}'.format(mean,var,x))

        acc = math.exp(-(x - mean)*(x - mean)/(2*var))/math.sqrt(2*math.pi*var)
        # print('acc:{}'.format(acc))
        return acc


    def calculate_prior(self, c):
        """ 计算c类的先验"""
        #c类的先验概率即为在训练集中该类记录所占的比例
        num = 0
        for i in self.y:
            if i == c:
                num += 1
        prior_c = num/len(self.y)
        # print(str(c)+'概率为：'+str(prior_c))
        return prior_c
        


    def classify(self, sample):
        """ 使用贝叶斯规则进行分类 P(Y|X) = P(X|Y)*P(Y)/P(X),或者 后验=可能性*先验/比例因子

        P(Y|X) - 后验是样本x属于y类的概率，假设x的特征值根据y和先验的分布分布。
        P(X|Y) - 给定类分布的数据X的可能性Y高斯分布（由_calculate_likelihood给出）
        P(Y)   - 先验（由_calculate_prior给出）
        P(X)   - 缩放后部以使其成为合适的概率分布。 此实现在此实现中被忽略，因为它不会影响样本最可能属于哪个类分布。

        将样本分类为导致最大P（Y | X）（后验）的类
        """
        posteriors = []
        # 浏览每一类的的list
        for i, c in enumerate(self.classes):
            # 将事后初始化为先验
            posterior = self.calculate_prior(c)
            # 贝叶斯假设（独立）：
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # 后验是先验和可能性的乘积（忽略比例因子）
            for feature_value, params in zip(sample, self.parameters[i]):
                # 给定y给出特征值分布的特征值的可能性；分别计算每个特征属性的的先验概率
                likelihood = self.calculate_likelihood(params["mean"], params["var"], feature_value)
                # print(params['mean'])
                # print(params['var'])
                # print(feature_value)
                posterior = posterior * likelihood
            posteriors.append(posterior)
            
        # 返回具有最大后验概率的类
        
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """ 预测X中样本的类标签 """
        y_pred = [self.classify(sample) for sample in X]
        return y_pred
