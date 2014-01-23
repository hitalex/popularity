#coding=utf8

"""
这里提出一种解决各个feature量纲不同的方法。
主要思想是：对于每个feature，对train set中的样本按照相似度进行排序，并将训练样本的序号作为其得分值。
最后，对于每个训练样本，计算其总的得分值，并按照总的分排序，然后按照此排序得到最近邻。

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
    dis = best_match_distance(vec1, vec2, 0)
    
    return dis
        
def weighted_vote(distance_comment_list, gamma):
    """ 对每个factor分别使用weighted vote进行popularity预测，最后再进行平均
    计算ts之间的距离，并将exp(-lambda * dis(t1, t2))作为weight进行投票
    Note: distance之间相差可能较大，所以需要先进行归一化
    """
    # normalize distance
    count = len(distance_comment_list)
    dis_list = [0] * count
    for i in range(count):
        topic_id, dis, num_comment = distance_comment_list[i]
        dis_list[i] = dis
        
    from sklearn.preprocessing import scale
    #dis_list = scale(dis_list, with_mean=False, with_std=True)
    dis_list = np.array(dis_list, float)
    if np.max(dis_list) > 0:
        dis_list /= np.max(dis_list)
    
    pred = 0
    Z = 0
    for i in range(count):
        topic_id, dis, num_comment = distance_comment_list[i]
        try:
            weight = math.exp(-gamma * dis_list[i])
        except OverflowError:
            print 'Error in math.exp: ', -gamma * dis_list[i]
            continue
            
        pred +=  weight * num_comment
        Z += weight
    
    pred /= Z
    
    return pred
    
def gen_score_list(topic_score, topic_popularity):
    """ 根据topic score dict和topic_popularity生成score list
    """
    # from topic_score to list, so it can be ordered
    score_list = [0] * len(topic_score)
    index = 0
    for topic_id in topic_score:
        level, target_comment_count, prediction_comment_count = topic_popularity[topic_id]
        score = topic_score[topic_id]
        ratio = target_comment_count * 1.0 / prediction_comment_count
        score_list[index] = [score, target_comment_count, level, ratio, topic_id]
        index += 1
    
    # 按照得分，从低到高排序
    score_list.sort(key=operator.itemgetter(0), reverse=False)
    return score_list

def score_ranking_knn(test_ins, train_set, k, gamma = 1):
    """ 按照score ranking的方法找到k近邻
    gamma_list: the scaling parameters of each dynamic factors for weighted vote
    """
    num_feature = len(test_ins[1]) # get the number of features
    # 初始化所有的topic的rank score
    # topic_id ==> (dis, rank score). Highly similar instances have large score
    topic_score = dict()
    topic_popularity = dict()    # topic_id ==> (level, comment_count)
    for train_topic_id, train_ins, level in train_set:
        topic_score[train_topic_id] = 0
        target_comment_count = train_ins[0][0]
        prediction_comment_count = train_ins[0][4]
        topic_popularity[train_topic_id] = (level, target_comment_count, prediction_comment_count)
        
    # 对于所有的feature，分别排序计算得分
    train_count = len(train_set)
    weighted_vote_pred = 0
    topic_score_list = [] # 每个feature都保存一份score list
    for findex in range(num_feature):
        #print 'Caculating score and rank for feature: ', findex
        topic_score_feature = dict(topic_score)
        distance_comment_list = [0] * train_count
        index = 0
        #import ipdb; ipdb.set_trace()
        for train_topic_id, train_ins, level in train_set:
            # 程序的瓶颈：计算两个ts的距离
            dis = get_instance_distance(test_ins, train_ins, findex)
            target_comment_count = topic_popularity[train_topic_id][1]
            distance_comment_list[index] = [train_topic_id, dis, target_comment_count]
            index += 1
            
        # 按照dis进行升序排序
        distance_comment_list.sort(key=operator.itemgetter(1), reverse=False)
        
        for i in range(train_count):
            topic_id = distance_comment_list[i][0]
            # 根据排名计算得分，距离越近，即相似度越大，分数越低
            topic_score_feature[topic_id] = i   # the score
        
        topic_score_list.append(topic_score_feature)
        score_list_feature = gen_score_list(topic_score_feature, topic_popularity)
        print 'Score list for feature: %d' % findex
        print score_list_feature[:k]
        # 在k近邻中查找
        weighted_vote_pred += weighted_vote(distance_comment_list[:k], gamma)
        
    weighted_vote_pred = weighted_vote_pred * 1.0 / num_feature

    for topic_id in topic_score:
        for findex in range(num_feature):
            topic_score[topic_id] += topic_score_list[findex][topic_id]
    score_list = gen_score_list(topic_score, topic_popularity)
    print '\nOverall score list: ', score_list[:k]
    
    knn_level = [0] * k
    knn_topic_id = [0] * k # 真正的k个近邻的topic id
    # 将k近邻的评论数进行加权平均，将score值作为权值
    weighted_num_comment = 0
    weighted_ratio = 0
    total_score = 0
    for i in range(k):
        score = score_list[i][0]
        comment_count = score_list[i][1]
        level = score_list[i][2]
        ratio = score_list[i][3]
        topic_id = score_list[i][4]
        
        knn_level[i] = level
        knn_topic_id[i] = topic_id
        
        total_score += score
        weighted_num_comment += (score * comment_count)
        weighted_ratio += (score * ratio)
    
    weighted_num_comment = weighted_num_comment * 1.0 / total_score
    weighted_ratio = weighted_ratio * 1.0 / total_score
    weighted_ratio_pred = test_ins[0][4] * weighted_ratio
    
    return knn_level, knn_topic_id, weighted_ratio_pred
