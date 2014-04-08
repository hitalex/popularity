#coding=utf8

"""
IPW算法：使用mutual knn方法确定自己的邻居，其他和instance_prior_weighting3相同
"""
import operator
import math

import numpy as np

from instance_prior_weighting3 import get_instance_distance, get_knn_level_list_old, confidence_score_prediction
from utils import smooth, my_min_max_scaler

def factor_knn(findex, topic_popularity, dataset, num_neigh):
    """ 对于每个dynamic factor，分别计算其knn邻居
    """
    num_level = 2
    # topic_id ==> set of knn neighbours' topic id
    factor_knn_graph = dict()
    total = len(dataset)
    for topic_id, ins, level in dataset:
        print 'Finding knn for topic: ', topic_id
        distance_comment_list = [0] * total
        index = 0
        for topic_id_other, ins_other, level_other in dataset:
            if topic_id_other == topic_id:
                continue
            # 程序的瓶颈：计算两个ts的距离
            dis = get_instance_distance(ins, ins_other, findex)
            if dis == 0:
                dis = 1e-6
            level                   = topic_popularity[topic_id_other][0]
            target_comment_count    = topic_popularity[topic_id_other][1]
            prediction_comment_count= topic_popularity[topic_id_other][2]
            ratio                   = topic_popularity[topic_id_other][3]
            
            distance_comment_list[index] = [topic_id_other, dis, prediction_comment_count, target_comment_count, level, ratio]
            index += 1
            
        distance_comment_list = distance_comment_list[:index]
        # 按照dis进行升序排序
        distance_comment_list.sort(key=operator.itemgetter(1), reverse=False)
        # 将所有的最短距离都记录
        # 需要确保knn_level_list中包括两类的样本
        knn_level_list, knn_list, level_count_list = get_knn_level_list_old(distance_comment_list, num_neigh, num_level)
        
        factor_knn_graph[topic_id] = set()
        k = len(knn_list)
        for i in range(k):
            neighbour_topic_id = knn_list[i][0]
            factor_knn_graph[topic_id].add(neighbour_topic_id)
    
    return factor_knn_graph
    
def create_mutual_knn_graph(dataset, num_neigh, num_factor, topic_popularity):
    """ 根据训练集和测试集，构建mutual knn图
    train_set, test_set: the whole data set
    num_neigh: number of neighbours
    """
    factor_knn_graph_list = [] # for each dynamic factor
    for findex in range(num_factor):
        print 'Caculating factor knn for factor: ', findex
        factor_knn_graph = factor_knn(findex, topic_popularity, dataset, num_neigh)
        factor_knn_graph_list.append(factor_knn_graph)
        
    # create mutual knn graph
    mutual_knn_graph_list = []
    for findex in range(num_factor):
        print 'Caculating factor knn for factor: ', findex
        mutual_knn_graph = factor_knn_graph_list[findex]
        for topic_id in mutual_knn_graph:
            # 检查topic_id的每一个knn邻居
            mutual_knn_set = set(mutual_knn_graph[topic_id])
            for topic_id_other in mutual_knn_graph[topic_id]:
                if not topic_id in mutual_knn_graph[topic_id_other]:
                    mutual_knn_set.remove(topic_id_other)
                    
            mutual_knn_graph[topic_id] = mutual_knn_set
            
        mutual_knn_graph_list.append(mutual_knn_graph)
    
    return mutual_knn_graph_list
    
def caculate_instance_prior_confidence_score(train_set, test_set, num_neigh, num_factor, num_level = 2):
    """ 在holdout dataset(或者训练集)中计算每个instance在每个dynamic factor的先验信心分数
    """
    # merge the train and test set
    dataset = list(train_set)
    dataset.extend(test_set)
    
    topic_popularity = dict()    # topic_id ==> (level, comment_count)
    for topic_id, ins, level in dataset:
        target_comment_count = ins[0][0]
        prediction_comment_count = ins[0][4]
        # ratio的值不小于1
        ratio = target_comment_count * 1.0 / prediction_comment_count
        topic_popularity[topic_id] = (level, target_comment_count, prediction_comment_count, ratio)
    
    print 'Creating mutual knn graphs...'
    mutual_knn_graph_list = create_mutual_knn_graph(dataset, num_neigh, num_factor, topic_popularity)
    #import ipdb; ipdb.set_trace()
    
    # train_topic_id ==> [i, j] for factor i on class j
    prior_score = dict()
    total = len(train_set)
    index = 0
    factor_correct_count = np.zeros((num_factor,), float)
    for train_topic_id, train_ins, true_level in train_set:
        print 'Topic id: %s, Iteration: %d, true level: %d' % (train_topic_id, index, true_level)
        # 记录评分矩阵
        score_matrix = np.zeros((num_factor, num_level))
        for findex in range(num_factor):
            # 使用原来的近邻挑选方法
            level_confidence_score, level_prior_score = factor_score_knn(findex, mutual_knn_graph_list[findex], train_topic_id, topic_popularity, num_level)
            level_confidence_score = smooth(level_confidence_score)
            score_matrix[findex, :] = level_confidence_score
        
        pred_level_list = [0] * num_factor
        num_correct = 0 # 得出正确结果的factor的个数
        for findex in range(num_factor):
            # predict based on confidence
            pred_level_list[findex] = pred_level = np.argmax(score_matrix[findex, :])
            print 'Factor %d: confidence = %r, prediction = %d' % (findex, score_matrix[findex, :], pred_level)
            if pred_level != true_level:
                pass
            else:
                num_correct += 1
                factor_correct_count[findex] += 1 # 每个factor预测正确的样本个数
        
        # 计算先验，满足两个要求
        # 每个instance都保存有某个factor对其分类结果的信息，如果分类正确，则权重大于1，如果分类错误，则小于1
        level_prior = np.ones((num_factor, )) # prior for classes(levels)
        
        # 在factor之间之间进行区别：例如如果只有一个factor预测正确，那么奖励会更多
        delta = 1.0
        if num_correct == 0:
            correct_reward = 1.0
        else:
            correct_reward = delta / num_correct
        
        rho = 3
        for findex in range(num_factor):
            # 预测两类的信心值之差
            diff_score = abs(score_matrix[findex, 0] - score_matrix[findex, 1])
            if pred_level_list[findex] != true_level:
                tmp = math.exp(-1 * rho * diff_score)
            else:
                tmp = math.exp(+1 * rho * diff_score)

            # 另外一层考虑：例如如果只有一个factor预测正确，则奖励会更多
            if pred_level_list[findex] == true_level:
                tmp *= math.exp(correct_reward)
            
            level_prior[findex] *= tmp
        
        #level_prior /= np.sum(level_prior)
        prior_score[train_topic_id] = level_prior
        print 'Instance level prior for %s: %r\n' % (train_topic_id, level_prior)
        
        index += 1
        #print 'Training acc of single factors:', factor_correct_count / total
    
    print 'Training acc of single factors:', factor_correct_count / total
    return topic_popularity, prior_score, mutual_knn_graph_list
    
def factor_score_knn(findex, mutual_knn_graph, topic_id, topic_popularity, num_level, prior_score = -1, gamma = 1):
    """ 计算每个topic的confidence score和level score
    """
    # 标记是否考虑先验信息
    with_prior_flag = isinstance(prior_score, dict)
    
    num_neighbour = len(mutual_knn_graph[topic_id])
    neighbour_topic_id = list(mutual_knn_graph[topic_id])
    
    level_confidence_score = np.zeros((num_level,), float)
    # TODO： 这里的weight的值很可能覆盖prior
    Z = 0
    level_prior_score = np.array([0] * num_level, float)
    # normalize the distance
    dis_list = [0] * num_neighbour
    for i in range(num_neighbour):
        dis_list[i] = 0 # Note: 暂不考虑样本距离带来的影响
    
    # use the min-max normalizer
    #dis_list = my_min_max_scaler(dis_list)
    #print 'Transformed distance list:', dis_list
    #import ipdb; ipdb.set_trace()
    Z = [0] * 2
    gamma = 1
    for i in range(num_neighbour):
        topic_id = neighbour_topic_id[i]
        #dis = knn_list[i][1]
        level = topic_popularity[topic_id][0]
        dis = dis_list[i] # 使用归一化的距离
        
        try:
            weight = math.exp(-gamma * dis)
        except OverflowError:
            print 'Error in math.exp: ', -gamma * dis_list[i]
            continue
        
        Z[level] += weight
        if with_prior_flag: # 如果已经传递了先验信息
            # 可能该topic_id为测试样本，所以不会出现在prior_score中
            if not topic_id in prior_score:
                continue
                
            #import ipdb; ipdb.set_trace()
            level_confidence_score[level] += weight
            # 计算每个instance在这个factor下的level prior score
            level_prior = prior_score[topic_id]
            level_prior_score[level] += (weight * level_prior[findex])
        else:
            level_confidence_score[level] += weight
    
    # normalize
    if sum(Z) > 0:
        level_confidence_score /= sum(Z)
    else:
        print 'Warning: %s dose not have any mutual knn neighbours.' % (topic_id)
        level_confidence_score[:] = 1/num_level
        
    # 在不同factor下的level confidence下加入level_prior_score信息
    if with_prior_flag:
        if Z[0] > 0:
            level_prior_score[0] /= Z[0]
        else:
            level_prior_score[0] = 0
            
        if Z[1] > 0:
            level_prior_score[1] /= Z[1]
        else:
            level_prior_score[1] = 0
        
        if np.sum(level_prior_score) > 0:
            # 归一化， 此时 level_prior_score 的作用和level_confidence_score相同，只不过包含了先验信息
            level_prior_score /= np.sum(level_prior_score)
        else:
            level_prior_score[:] = 1/num_level
        
    #print 'Level confidence score:', level_confidence_score
    
    return level_confidence_score, level_prior_score
    
def weighted_vote_instance_prior(test_topic_id, mutual_knn_graph_list, num_factor, topic_popularity, prior_score = -1, gamma = 1, num_level = 2):
    """ 按照score ranking的方法找到k近邻
    gamma_list: the scaling parameters of each dynamic factors for weighted vote
    """
    #import ipdb; ipdb.set_trace()
    #注：分别在不同的dynamic factor中查找最近邻，然后将这些最近邻组合起来投票
    factor_confidence_score = np.zeros((num_factor, num_level))
    level_prior_score = np.zeros((num_factor, num_level))
    for findex in range(num_factor):
        #print 'Caculating score and rank for feature: ', findex
        factor_confidence_score[findex, :] , level_prior_score[findex, :] = factor_score_knn(findex, mutual_knn_graph_list[findex], \
            test_topic_id, topic_popularity, num_level, prior_score, gamma = 1)
    
    print 'Factor confidence:\n', factor_confidence_score
    print 'prediction: ', np.sum(factor_confidence_score, axis=0)
    
    print 'After adding priors:'
    print 'Factor confidence:\n', level_prior_score
    print 'prediction: ', np.sum(level_prior_score, axis=0)
    
    # 存储每个factor的预测结果
    factor_prediction = [-1] * num_factor
    for i in range(num_factor):
        pred = np.argmax(factor_confidence_score[i, :])
        factor_prediction[i] = pred
    
    prediction_level, confidence_score = confidence_score_prediction(level_prior_score)
    #print 'Overall prediction: %d with confidence: %f' % (prediction_level, confidence_score)
    
    return [prediction_level], '', 0, factor_prediction
