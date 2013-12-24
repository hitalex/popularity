#coding=utf8

"""
Baseline methods of popularity prediction, which includes:
1) S-H model
2) Multivariate Linear(ML) model
3) ARIMA model(time series)
4) simple KNN method
5) Perhaps the method proposed in CIKM'13
"""

import numpy as np
import numpy.linalg as LA
from sklearn import linear_model

def insert_neighbour(knn_list, vec):
    """ 将vec插入到knn list中
    """
    dis = vec[0]
    flag = False
    k = len(knn_list)
    for i in range(k):
        if dis < knn_list[i][0]:
            flag = True
            break
    
    if flag:
        knn_list.insert(i, vec)
        if len(knn_list) > k:
            knn_list.pop()
        return i
    else:
        return k
        
def ARIMA_model(train_set, test_set, num_feature, alpha):
    """ ARIMA model
    Note: ARIMA模型没有训练过程，所以这里的训练集用不上
    问题：如何确定p, q, d
    """
    import statsmodels.api as sm
    
    x_test, comment_true, y_true = MLR_extract_feature(test_set, num_feature)
    index = 0
    for topic_id, comment_count_feature, cat in test_set:
        #arima_model = tsa.arima_model.ARIMA(x_test[index, :], ())
        pass
    
    return
    
def LSM_method(train_set, test_set, num_feature, alpha):
    """ NIPS'13: A latent source model for nonparametric time series classification
    """
    from utils import transform_ts
    x_train, comment_cnt_train, y_train = MLR_extract_feature(train_set, num_feature)
    train_cnt = len(train_set)
    for i in range(train_cnt):
        x_train[i, :] = transform_ts(x_train[i, :])
        
    x_test, comment_cnt_true, y_true = MLR_extract_feature(test_set, num_feature)
    
    test_cnt = len(x_test)
    for i in range(test_cnt):
        s = x_test[i, :] = transform_ts(x_test[i, :])
        
    
def Bao_method(train_set, test_set, alpha):
    """ WWW'13: Popularity Prediction in Microblogging Network: A Case Study on Sina Weibo
    注：文章中实际上提出两个模型，分别使用了link density和diffusion depth作为特征，进行回归
    """
    #import ipdb; ipdb.set_trace()
    from math import log, exp
    train_cnt = len(train_set)
    x_train = np.zeros((train_cnt, 2), float)
    y_train = np.zeros((train_cnt,) , float)
    index = 0
    for topic_id, prediction_comment_count, target_comment_count, \
            prediction_link_density, prediction_diffusion_depth, cat in train_set:
        
        x_train[index, 0] = log(prediction_comment_count)
        #x_train[index, 1] = log(prediction_link_density)
        x_train[index, 1] = prediction_diffusion_depth
        y_train[index] = log(target_comment_count)
        index += 1
        
    clf = linear_model.LinearRegression(fit_intercept=True).fit(x_train, y_train)
    params = clf.coef_
    print 'Bao\'s method coef: %r, intercept: %r' % (params, clf.intercept_)
    
    
    test_cnt = len(test_set)
    x_test = np.zeros((test_cnt, 2), float)
    y_test = np.zeros((test_cnt,) , float)
    index = 0
    for topic_id, prediction_comment_count, target_comment_count, \
            prediction_link_density, prediction_diffusion_depth, cat in test_set:
        
        x_test[index, 0] = log(prediction_comment_count)
        #x_test[index, 1] = log(prediction_link_density)
        x_test[index, 1] = prediction_diffusion_depth
        y_test[index] = log(target_comment_count)
        index += 1
        
    # predict with the trained model
    comment_pred = clf.predict(x_test)
    y_true = [0] * test_cnt
    y_pred = [0] * test_cnt
    comment_true = [0] * test_cnt
    for i in range(test_cnt):
        comment_true[i] = test_set[i][2]
        comment_pred[i] = exp(comment_pred[i])
        y_true[i] = test_set[i][5]
        
        if comment_pred[i] > test_set[i][1] * alpha:
            cat = 1
        else:
            cat = 0
        y_pred[i] = cat
    
    return y_true, y_pred, comment_true, comment_pred

def knn_method(train_set, test_set, k, num_feature, alpha):
    """ Finding the k-nearest neighbour and get a weighted vote
    k: the number of nearest neighbours
    """
    test_cnt = len(test_set)
    y_true = [0] * test_cnt
    y_pred = [0] * test_cnt
    comment_true = [0] * test_cnt
    comment_pred = [0] * test_cnt
    index = 0
    for test_topic_id, test_comment_count_feature, test_cat in test_set:
        test_ins = np.array(test_comment_count_feature[1:num_feature+1], float)
        knn_list = [(float('inf'), '', 0, 0)] * k # [distance, topic_id, target comment count, cat]
        for topic_id, comment_count_feature, cat in train_set:
            target_comment_count = comment_count_feature[0][0]
            ins = np.array(comment_count_feature[1:num_feature+1], float)
            dis = LA.norm(test_ins - ins)
            if dis == 0:
                dis = 1e-6
            vec = [dis, topic_id, target_comment_count]
            insert_neighbour(knn_list, vec)
            
        # caculate weighted comment
        y_true[index] = test_cat
        comment_true[index] = test_comment_count_feature[0][0]
        weighted_comment_cnt = 0
        weight_sum = 0
        for i in range(k):
            w = 1.0 / knn_list[i][0]
            weighted_comment_cnt +=  w * knn_list[i][2]
            weight_sum += w
            
        comment_pred[index] = weighted_comment_cnt / weight_sum
        if comment_pred[index] > alpha * test_comment_count_feature[0][4]:
            y_pred[index] = 1
        else:
            y_pred[index] = 0
        
        index += 1
        
    return y_true, y_pred, comment_true, comment_pred
    
def MLR_extract_feature(dataset, num_feature):
    """ Extract feature for ML model
    """
    count = len(dataset)
    x = np.zeros((count, num_feature), float) # each train/test sample for each row
    y = [0] * count
    comment_cnt = [0] * count
    index = 0
    for topic_id, ins_feature, true_level in dataset:
        target_comment_cnt = ins_feature[0][0]
        comment_cnt[index] = target_comment_cnt
        x[index, :] = ins_feature[1:num_feature+1]
        y[index] = true_level
            
        index += 1
    
    return x, comment_cnt, y

def MLR_model(train_set, test_set, num_feature, alpha):
    """ MLR model, i.e. time series regression
    Simply use interval level comment count as features
    """
    x_train, comment_cnt_train, y_train = MLR_extract_feature(train_set, num_feature)
    
    # fit a multivariate regressor using scikit-learn OLS
    from sklearn import linear_model
    # http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    # without intercept
    clf = linear_model.LinearRegression().fit(x_train, y_train)
    params = clf.coef_
    print 'MLR coef: %r, intercept: %r' % (params, clf.intercept_)

    # using the model to predict
    x_test, comment_true, y_true = MLR_extract_feature(test_set, num_feature)
    
    comment_pred = clf.predict(x_test)
    
    test_cnt = len(x_test)
    y_pred = [0] * test_cnt
    
    for i in range(test_cnt):
        prediction_comment_cnt = test_set[i][1][0][4]
        if comment_pred[i] * 1.0 / prediction_comment_cnt > alpha:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    
    return y_true, y_pred, comment_true, comment_pred

def ML_extract_feature(dataset, num_feature):
    """ Extract feature for ML model
    """
    count = len(dataset)
    x = np.zeros((count, num_feature - 1), float) # each train/test sample for each row
    y = [0] * count
    index = 0
    for topic_id, ins_feature, true_level in dataset:
        target_comment_cnt = ins_feature[0][0]
        y[index] = target_comment_cnt
        for i in range(1, num_feature):
            # use comment increase as features
            inc = ins_feature[i+1] - ins_feature[i]
            x[index, i-1] = inc * 1.0 / target_comment_cnt
            
        index += 1
    
    return x, y

def ML_model(train_set, test_set, num_feature, alpha):
    """ Multivariate Linear(ML) model
    Use the comment increase rate as features and formulate a regression problem
    """
    #import ipdb; ipdb.set_trace()
    x_train, comment_cnt_train = ML_extract_feature(train_set, num_feature)
    y_train = [1] * len(x_train)
    
    # fit a multivariate regressor using scikit-learn OLS
    # http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    # without intercept
    clf = linear_model.LinearRegression(fit_intercept=False).fit(x_train, y_train)
    params = clf.coef_
    print 'ML coef:', params
    
    """
    import statsmodels.api as sm
    #x_train = sm.add_constant(x_train) # add intercept
    model = sm.OLS(y_train, x_train).fit()
    params = model.params
    print model.params
    """
    # using the model to predict
    x_test, comment_true = ML_extract_feature(test_set, num_feature)
    test_cnt = len(x_test)
    comment_pred = [0] * test_cnt
    y_pred = [0] * test_cnt
    y_true = [0] * test_cnt
    
    for i in range(test_cnt):
        sample = x_test[i, :]
        comment_pred[i] = np.dot(sample, params) * comment_true[i]
        
        y_true[i] = test_set[i][2]
        prediction_comment_cnt = test_set[i][1][0][4]
        if comment_pred[i] * 1.0 / prediction_comment_cnt > alpha:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    
    return y_true, y_pred, comment_true, comment_pred

def plot_SH_result(prediction_point_cnt, comment_true, comment_pred):
    """ 将SH模型的预测结果展示出来
    """
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    #ax.plot(prediction_point_cnt, c='r')
    ax.plot(comment_true, ls='-.', c='b')
    ax.plot(comment_pred, c='g')
    
    plt.show()
    
    cnt = len(prediction_point_cnt)
    delta = [comment_true[i] - prediction_point_cnt[i] for i in range(cnt)]
    print 'Delta number of comments:', delta
    
def SH_model(train_set, test_set, alpha):
    """ Szabo-Huberman model assumes there is a high linear correlation between
    the log-transformed early and future popularity of online content up to a
    normally distributed noise.
    """
    # 从训练集中计算线性相关的系数beta
    ratio = [0] * len(train_set)
    index = 0
    for topic_id, ins_feature, true_level in train_set:
        target_date_comment_cnt = ins_feature[0][0]
        prediction_date_point = ins_feature[0][1]
        target_date_point = ins_feature[0][2]
        topic_id = ins_feature[0][3]
        prediction_date_comment_cnt = ins_feature[0][4]
        
        ratio[index] = prediction_date_comment_cnt*1.0/target_date_comment_cnt
        index += 1
    
    ratio = ratio[:index]
    ratio = np.array(ratio, float)
    beta = sum(ratio) / sum(ratio ** 2)
    print 'S-H model linear coef: ', beta
    
    test_cnt = len(test_set)
    prediction_date_comment_cnt = [0] * test_cnt
    comment_true = [0] * test_cnt
    comment_pred = [0] * test_cnt
    index = 0
    y_true = [0] * test_cnt
    y_pred = [0] * test_cnt
    # prediction over the test set
    for topic_id, ins_feature, true_level in test_set:
        prediction_date_point = ins_feature[0][1]
        comment_true[index] = ins_feature[0][0]
        
        prediction_date_comment_cnt[index] = ins_feature[0][4]
        comment_pred[index] = beta * prediction_date_comment_cnt[index]
        
        y_true[index] = true_level
        if beta > alpha:
            y_pred[index] = 1
        else:
            y_pred[index] = 0
        
        index += 1
    
    #print 'prediction_date_comment_cnt:', prediction_point_cnt
    #print 'comment_true:', comment_true
    #print 'comment_pred:', comment_pred
    
    #plot_SH_result(prediction_date_comment_cnt, comment_true, comment_pred)
    
    return y_true, y_pred, comment_true, comment_pred
