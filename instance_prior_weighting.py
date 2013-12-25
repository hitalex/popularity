#coding=utf8

"""
这里采用的是weighted vote方法：分别在每个factor下寻找最近邻（可能是k近邻），然后进行weighted vote。
具体来说：
1，考察每个factor的先验分类信心值
2，计算每个factor对于每个类的分类信心值
    2.1 对于近邻列表，分别计算每个vote的weight值，按照distance计算（如果距离都相同，则都为1）
    2.2 分别对于两个类别，统计每个类别的总weight数，即得到此factor对于每个类别的信心值（注意此时无需判断is_trusted）
3, 对于所有的factor，分别对两类的信心值相加，则得到总的信心值。取两者之间较高的那个。

注意：这里其实是一种投票的思想，而且默认所有的feature的贡献是一样的。
"""
import numpy as np

import operator
import math

from ts_distance import DTW_distance, best_match_distance

def get_instance_distance(test_ins, train_ins, findex):
    """ Caculate DTW distance between two instances
    findex: feature index to be used
    Note: 只能处理一个feature
    """
    # 准备两个vector，作为DTW函数的输入
    pindex1 = test_ins[0][1]    # prediction point of instance 1
    pindex2 = len(train_ins)
    
    # instance的第0个feature是无用的，所以从1开始
    vec1 = [0] * (pindex1-1)
    for i in range(1, pindex1):
        vec1[i-1] = test_ins[i][findex]
        
    vec2 = [0] * (pindex2-1)
    for i in range(1, pindex2):
        vec2[i-1] = train_ins[i][findex]
        
    #dis = DTW_distance(vec1, vec2)
    dis = best_match_distance(vec1, vec2, 20)
    
    return dis
    
def check_convergence_plot(data):
    """ 将prior中的所有变量的变化历史画出来
    """
    import matplotlib.pyplot as plt
    num_var, total = data.shape
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    ax = plt.subplot(111)
    
    for i in range(num_var):
        ax.plot(data[i, :], c=colors[i])
        
    ax.set_ylim((0,1))
    
    plt.show()
    
def caculate_prior_confidence_score(train_set, k, num_level = 2):
    """ 在holdout dataset(或者训练集)中计算每个dynamic factor的先验信心分数
    """
    import random
    num_factor = len(train_set[0][1][1]) # get the number of factors
    topic_popularity = dict()    # topic_id ==> (level, comment_count)
    for train_topic_id, train_ins, level in train_set:
        target_comment_count = train_ins[0][0]
        prediction_comment_count = train_ins[0][4]
        ratio = target_comment_count * 1.0 / prediction_comment_count
        topic_popularity[train_topic_id] = (level, target_comment_count, prediction_comment_count, ratio)
    
    # prior_score[i, j]: for factor i, P(true=j | pred=j)
    prior_score = np.ones((num_factor, num_level))
    total = len(train_set)
    # 检查是否收敛
    score_history = np.zeros((num_factor * num_level, total), float)
    index = 0
    for topic_id, ins, true_level in train_set:
        if random.random() < 0.9:
            continue
        print 'Iteration: ', index
        for i in range(num_level):
            Z = 0
            for findex in range(num_factor):
                level_confidence_score = factor_score_knn(findex, ins, train_set, topic_popularity, k, num_level, gamma = 1)
                # predict based on confidence
                pred_level = np.argmax(level_confidence_score)
                # 此处参考Boosting算法: Boosting是对instance进行weighting，而这里是对分类器计算prior
                err = 1 - level_confidence_score[i]
                if err < 0.2:
                    err = 0.2
                elif err > 0.8:
                    err = 0.8
                    
                alpha = 0.5 * math.log((1-err) / err)
                if pred_level == true_level:
                    weight = math.exp(-1 * alpha)
                else:
                    weight = math.exp(alpha)
                    
                prior_score[findex, i] *= weight
                Z += prior_score[findex, i]
                
            # normlization
            prior_score[: , i] /= Z
        
        print 'Current prior info: ', prior_score
        #import ipdb; ipdb.set_trace()
        # 记录历史prior数据
        for i in range(num_factor):
            for j in range(num_level):
                row_index = i * num_level + j
                score_history[row_index, index] = prior_score[i, j]
        
        index += 1
        
    score_history = score_history[:, index]
    
    #check_convergence_plot(score_history)
    
    return prior_score
        
def weighted_vote(knn_list, num_level, gamma):
    """ 对每个factor分别使用weighted vote进行popularity预测，最后再进行平均
    计算ts之间的距离，并将exp(-lambda * dis(t1, t2))作为weight进行投票
    Note: distance之间相差可能较大，所以需要先进行归一化
    """
    # normalize distance
    count = len(knn_list)
    confidence_score = np.array([0] * num_level, float)
    Z = 0
    for i in range(count):
        dis = knn_list[i][1]
        level = knn_list[i][4]
        try:
            weight = math.exp(-gamma * dis)
        except OverflowError:
            print 'Error in math.exp: ', -gamma * dis_list[i]
            continue
            
        confidence_score[level] +=  weight
        Z += weight
    
    if Z > 0:
        confidence_score /= Z
    
    return confidence_score
    
def is_trusted(nn_level_list, num_level=2, majority_threshold=0.66):
    """ 判断一个level的列表是否值得相信
    majority_threshold: 其中一个level的比例大于此值，则值得相信
    """
    level_count = [0] * num_level
    for level in nn_level_list:
        level_count[level] += 1
        
    total = len(nn_level_list)
    for i in range(num_level):
        level_count[i] = level_count[i] * 1.0 / total
        if level_count[i] >= majority_threshold:
            return True
            
    return False
    
def get_knn_level_list(distance_comment_list, k, level_count):
    """ 获取k近邻的level标签
    按照如下方式：首先获取最近邻，如果最近邻数超过k个，则返回全部的最近邻；
    如果最近邻数不足k个（可能有相同距离），则考虑次近邻；
    """
    knn_dis = [0] * k
    knn_dis[0] = distance_comment_list[0][1]
    knn_dis_count = [0] * k
    current_k_index = 0
    total = len(distance_comment_list)
    nn_level_list = []
    for i in range(1, total):
        dis = distance_comment_list[i][1]
        if knn_dis[current_k_index] == dis:
            continue
        
        knn_dis_count[current_k_index] = i
        current_k_index += 1
        if current_k_index >= k:
            break
        else:
            knn_dis[current_k_index] = dis
            
    total = knn_dis_count[0]
    i = 0
    while i < k:
        total = knn_dis_count[i]
        if total >= k:
            break
        i += 1
        
    knn_level_list = [0] * total
    level_count_list = [0] * level_count
    for i in range(total):
        level = distance_comment_list[i][4]
        knn_level_list[i] = level
        level_count_list[level] += 1
        
    return knn_level_list, distance_comment_list[:total], level_count_list
    
def confidence_score_prediction(factor_confidence_score):
    """ 根据每个factor在每个类别的信心值，作出预测和信心值
    """
    num_factor, num_level = factor_confidence_score.shape
    max_score = 0
    prediction = -1
    for i in range(num_level):
        score = 0
        for j in range(num_factor):
            score += factor_confidence_score[j, i]
            
        if score > max_score:
            max_score = score
            prediction = i
    
    max_score /= num_factor
    return prediction, max_score
    
def factor_score_knn(findex, test_ins, train_set, topic_popularity, k, num_level, gamma):
    """ 针对某一个factor查找其knn
    """
    test_topic_id = test_ins[0][3]
    train_count = len(train_set)
    distance_comment_list = [0] * train_count
    index = 0
    #import ipdb, ipdb.set_trace()
    for train_topic_id, train_ins, level in train_set:
        if train_topic_id == test_topic_id:
            continue
        # 程序的瓶颈：计算两个ts的距离
        dis = get_instance_distance(test_ins, train_ins, findex)
        if dis == 0:
            dis = 1e-6
        level                   = topic_popularity[train_topic_id][0]
        target_comment_count    = topic_popularity[train_topic_id][1]
        prediction_comment_count= topic_popularity[train_topic_id][2]
        ratio                   = topic_popularity[train_topic_id][3]
        
        distance_comment_list[index] = [train_topic_id, dis, prediction_comment_count, target_comment_count, level, ratio]
        index += 1
        
    distance_comment_list = distance_comment_list[:index]
    # 按照dis进行升序排序
    distance_comment_list.sort(key=operator.itemgetter(1), reverse=False)
    # 将所有的最短距离都记录
    knn_level_list, knn_list, level_count_list = get_knn_level_list(distance_comment_list, k, num_level)
    #print 'kNN list for factor: ', findex
    #print knn_list
    
    level_confidence_score = weighted_vote(knn_list, num_level, gamma)
    
    return level_confidence_score

def score_ranking_knn(test_ins, train_set, k, prior_score, gamma = 1):
    """ 按照score ranking的方法找到k近邻
    gamma_list: the scaling parameters of each dynamic factors for weighted vote
    """
    num_factor = len(test_ins[1]) # get the number of factors
    num_level = 2

    topic_popularity = dict()    # topic_id ==> (level, comment_count)
    for train_topic_id, train_ins, level in train_set:
        target_comment_count = train_ins[0][0]
        prediction_comment_count = train_ins[0][4]
        ratio = target_comment_count * 1.0 / prediction_comment_count
        topic_popularity[train_topic_id] = (level, target_comment_count, prediction_comment_count, ratio)
        
    # 对于所有的feature，分别排序计算得分
    train_count = len(train_set)
    #注：分别在不同的dynamic factor中查找最近邻，然后将这些最近邻组合起来投票
    knn_list_all = []
    factor_confidence_score = np.zeros((num_factor, num_level))
    for findex in range(num_factor):
        #print 'Caculating score and rank for feature: ', findex
        factor_confidence_score[findex, :]  = factor_score_knn(findex, test_ins, train_set, topic_popularity, k, num_level, gamma)

    #print '\nOverall score list: ', knn_list_all
    print 'Classification confidence score:', factor_confidence_score
    # add prior confidence score
    for findex in range(num_factor):
        for i in range(num_level):
            factor_confidence_score[findex, i] *= prior_score[findex, i]
    
    print 'After adding prior confidence score: ', factor_confidence_score
    
    prediction_level, confidence_score = confidence_score_prediction(factor_confidence_score)
    print 'Overall prediction: ', np.sum(factor_confidence_score, axis=0)
    #print 'Overall prediction: %d with confidence: %f' % (prediction_level, confidence_score)
    
    return [prediction_level], '', 0
