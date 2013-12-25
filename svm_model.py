#coding=utf8

"""
由于在使用非参数化方法时遇到无法融合模型的困难，这里将尝试采用feature-based方法。
具体来说：对于不同的factor，将它们的变化序列作为feature。这样，如果有m个factor，每个factor有n个
采样时间，那么svm模型处理的特征就有：m*n个
注意：由于要保证训练集和测试集特征数目的统一性，所以训练集中那些prediction date之后的信息便无法使用

Update: 在使用了很多模型之后，二分类的准确率也只是在70%左右。
"""

import numpy as np
from sklearn import svm

def prepare_dataset(dataset, num_factor, num_interval):
    """ prepare a dataset for train and test
    """
    num_feature = num_factor * num_interval
    count = len(dataset)
    x = np.zeros((count, num_feature), float)
    y = np.zeros((count,), int)
    
    index = 0
    for topic_id, ins, level in dataset:
        feature = np.zeros((num_feature,), float)
        for i in range(num_factor):
            for j in range(num_interval):
                findex = i * num_factor + j
                feature[findex] = ins[j+1][i]
                
        x[index, :] = feature
        y[index] = level
        index += 1
    
    return x, y

def svm_model(train_set, test_set):
    """ svm model
    """
    #import ipdb; ipdb.set_trace()
    num_factor = len(train_set[0][1][1])
    num_interval = train_set[0][1][0][1]       # prediction data point
    
    x_train, y_train = prepare_dataset(train_set, num_factor, num_interval)
    x_test, y_test = prepare_dataset(test_set, num_factor, num_interval)
    # train a svm model
    model = svm.SVC().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    return y_test, y_pred
