#coding=utf8

"""
这里采用的是weighted vote方法：分别在每个factor下寻找最近邻（可能是k近邻），然后进行weighted vote。
具体来说：
1，考察每个factor的先验分类信心值
   这里还是采用近邻的方法：对于每个训练样本，记录在使用近邻法进行分类时，每个factor在每类的分类信心。
      
2，计算每个factor对于每个类的分类信心值
    2.1 对于近邻列表，分别计算每个vote的weight值，按照distance计算（如果距离都相同，则都为1）
    2.2 分别对于两个类别，统计每个类别的总weight数，即得到此factor对于每个类别的信心值（注意此时无需判断is_trusted）
3, 对于所有的factor，分别对两类的信心值相加，则得到总的信心值。取两者之间较高的那个。
"""
import numpy as np

import operator
import math

from ts_distance import DTW_distance, best_match_distance
from utils import smooth

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
    
def caculate_instance_prior_confidence_score(train_set, k, num_level = 2):
    """ 在holdout dataset(或者训练集)中计算每个instance在每个dynamic factor的先验信心分数
    """
    import random
    num_factor = len(train_set[0][1][1]) # get the number of factors
    topic_popularity = dict()    # topic_id ==> (level, comment_count)
    for train_topic_id, train_ins, level in train_set:
        target_comment_count = train_ins[0][0]
        prediction_comment_count = train_ins[0][4]
        ratio = target_comment_count * 1.0 / prediction_comment_count
        topic_popularity[train_topic_id] = (level, target_comment_count, prediction_comment_count, ratio)
    
    # topic_id ==> [i, j] for factor i on class j
    prior_score = dict()
    total = len(train_set)
    index = 0
    factor_correct_count = np.zeros((num_factor,), float)
    for topic_id, ins, true_level in train_set:
        print 'Topic id: %s, Iteration: %d' % (topic_id, index)
        # 记录评分矩阵
        score_matrix = np.zeros((num_factor, num_level))
        for findex in range(num_factor):
            level_confidence_score = factor_score_knn(findex, ins, train_set, topic_popularity, k, num_level)
            level_confidence_score = smooth(level_confidence_score)
            score_matrix[findex, :] = level_confidence_score
        # 计算先验，满足两个要求
        level_prior = np.ones((num_factor, num_level)) # prior for classes(levels)
        pred_level_list = [0] * num_factor
        num_correct = 0 # 得出正确结果的factor的个数
        print 'Topic %s, true level: %d' % (topic_id, true_level)
        for findex in range(num_factor):
            # predict based on confidence
            pred_level_list[findex] = pred_level = np.argmax(score_matrix[findex, :])
            print 'Factor %d prediction: %d' % (findex, pred_level)
            if pred_level != true_level:
                pass
            else:
                num_correct += 1
                factor_correct_count[findex] += 1

        # 最简单的情形：直接使用score_matrix来作为factor信心值
        level_prior = score_matrix
        # 在factor之间之间进行区别：例如如果只有一个factor预测正确，那么奖励会更多
        rho = 2.0
        for findex in range(num_factor):
            if pred_level_list[findex] != true_level:
                the_other_level = 1 - true_level
                level_prior[findex, the_other_level] *= math.exp(-1 * 1.0 * rho / (num_factor-num_correct))
            else:
                level_prior[findex, true_level] *= math.exp(+1 * 1.0 * rho / num_correct)
                
            # normalize
            level_prior[findex, :] /= np.sum(level_prior[findex, :])
        
        prior_score[topic_id] = level_prior
        print 'Instance level prior for %s: %r' % (topic_id, level_prior)
        
        index += 1
        #print 'Training acc of single factors:', factor_correct_count / total
    
    print 'Training acc of single factors:', factor_correct_count / total
    return prior_score
    
def get_knn_level_list(distance_comment_list, k, level_count):
    """ 获取k近邻的level标签
    注意：有可能同一个距离有多个近邻，这时考虑的其实是前k个距离的所有近邻（与之前的计算方法不同）
    这样做的好处是：统一了计算方法，但最终的得到的近邻数可能大于k
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
            
    total = knn_dis_count[-1]

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
    
def factor_score_knn(findex, test_ins, train_set, topic_popularity, k, num_level, prior_score = -1, gamma = 1):
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
    print 'kNN list for factor: ', findex
    print knn_list
    
    num_neighbour = len(knn_list)
    level_confidence_score = np.zeros((num_level,), float)
    # TODO： 这里的weight的值很可能覆盖prior
    # 标记是否考虑先验信息
    with_prior_flag = isinstance(prior_score, dict)
    Z = 0
    for i in range(num_neighbour):
        topic_id = knn_list[i][0]
        dis = knn_list[i][1]
        level = knn_list[i][4]
        
        try:
            weight = math.exp(-gamma * dis)
        except OverflowError:
            print 'Error in math.exp: ', -gamma * dis_list[i]
            continue
        
        Z += weight
        if with_prior_flag: # 如果已经传递了先验信息
            level_prior = prior_score[topic_id]
            # 近邻的本身的level在这里体现：其实考虑了每个factor对于每类的信心值
            # 当前的近邻的标签是level，所以只选用了level_prior[findex, level]
            level_confidence_score[level] += (weight * level_prior[findex, level])
        else:
            level_confidence_score[level] += weight
    
    # normalize
    level_confidence_score /= np.sum(level_confidence_score)
    
    return level_confidence_score

def weighted_vote_instance_prior(test_ins, train_set, k, prior_score = -1, gamma = 1):
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
        
    #import ipdb; ipdb.set_trace()
    # 对于所有的feature，分别排序计算得分
    train_count = len(train_set)
    #注：分别在不同的dynamic factor中查找最近邻，然后将这些最近邻组合起来投票
    knn_list_all = []
    factor_confidence_score = np.zeros((num_factor, num_level))
    for findex in range(num_factor):
        #print 'Caculating score and rank for feature: ', findex
        factor_confidence_score[findex, :] = factor_score_knn(findex, test_ins, \
            train_set, topic_popularity, k, num_level, prior_score, gamma)

    #print '\nOverall score list: ', knn_list_all
    print 'Factor confidence:\n', factor_confidence_score
    print 'prediction: ', np.sum(factor_confidence_score, axis=0)
    
    prediction_level, confidence_score = confidence_score_prediction(factor_confidence_score)
    #print 'Overall prediction: %d with confidence: %f' % (prediction_level, confidence_score)
    
    return [prediction_level], '', 0
