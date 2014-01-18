#coding=utf8

"""
实现The ICDM 2013 method: Adjusted Confidence Score

Notes:
1, 这里将对之前的dataset进行转换，使用新的数据存储方式
2，本方法中要求计算1NN，但是有时会存在多个最近邻，所以这里打算先进行投票，然后最近距离采用均值
"""

import numpy as np
import pickle
import math

from ts_distance import DTW_distance, best_match_distance

class Instance(object):
    """ 将Python的类当作结构体使用
    """
    def __init__(self, topic_id, prediction_date_point, target_date_point, \
            prediction_comment_count, target_comment_count, feature, true_level):
        self.topic_id = topic_id
        self.prediction_date_point = prediction_date_point
        self.target_date_point = target_date_point
        self.prediction_comment_count = prediction_comment_count
        self.target_comment_count = target_comment_count
        self.feature = feature
        self.true_level = true_level
    
def get_query_samples(holdout_dataset, percent=0.3):
    """ Get a random set from holdout dataset
    percent: percentage of random samples from holdout dataset
    """
    return holdout_dataset
    
def majority_vote(NN_list, p):
    """ majority vote of each class
    """
    votes = [0] * p
    count = len(NN_list)
    for i in range(NN_list):
        level = NN_list[i].true_level
        votes[level] += 1
        
    return votes.index(max(votes))
    
def prob_density_estimation(NN_true, NN_false, target_dis, h):
    """ Use the non-parametric parzen window to evaluate probilities density
    """
    n = len(NN_true)
    hn = h / math.sqrt(h)
    
    
def One_NN_search(qs, holdout_dataset, dim_index, p):
    """ One nearest neighbour search in dataset
    qs: the query set
    holdout: holdout dataset
    dim_index: which dim is to search
    q: number of classes

    Note: 因为可能会找出多个最近邻，所以在这里会先进行投票，假如得票最多的是第i类，那么
    dist的值则由第i类中所有的近邻的dist进行平均得到
    """
    query_count = len(qs)
    NN_labels = [-1] * query_count
    NN_dists = [-1] * query_count
    
    total = len(holdout_dataset)
    for i in range(query_count):
        min_dist = float('inf')
        NN_list = []
        for j in range(total):
            if qs[i].topic_id == holdout_dataset[j].topic_id:
                continue
            prediction_date_point = qs[i].prediction_date_point
            vec1 = qs[i].feature[dim_index, :prediction_date_point+1]
            vec2 = holdout_dataset[j].feature[dim_index, :]
            dis = best_match_distance(vec1, vec2, 20)
            if dis < min_dist:
                NN_list = [holdout_dataset[j]]
                min_dist = dis
            elif dis == min_dist:
                # 可能有多个最近邻
                NN_list.append(holdout_dataset[j])
        
        assert(len(NN_list) > 0)
        if len(NN_list) == 1:
            ins = NN_list[0]
            NN_labels.append(ins.true_level)
        else:
            label = majority_vote(NN_list, p)
            NN_labels.append(label)
            
        NN_dists.append(min_dist)
        
    return NN_labels, NN_dists
    
def caculate_precision(NN_labels, NN_dists, qs, class_index):
    """ Caculate precision and NN_true, NN_false for a specific class
    """
    count = len(NN_labels)
    
    correct = 0 # number of correct guess
    total = 0   # number of guess for class_index
    NN_true = []
    NN_false = []
    for i in range(count):
        if NN_labels[i] != class_index:
            continue
        total += 1
        true_label = qs[i].true_label
        if NN_labels[i] == true_label:
            NN_true.append(NN_dists[i])
            correct += 1
        else:
            NN_false.append(NN_dists[i])
            
    precision = correct * 1.0 / total
    
    return NN_true, NN_false, precision
    
def caculate_confidence_score(holdout_dataset, p, m):
    """ Learning the confidence score
    p: number of classes
    m: number of dimensions
    
    Output:
    C_matrix: shape is (p, m), the confidence matrix
    DN_matrix: the distributions nearest neighbour distances
    """
    qs = get_query_samples(holdout_dataset, percent)
    C_matrix = np.zeros((m, p), dtype=float)
    DN_matrix = np.zeros((m, p), dtype=object)
    
    for i in range(m):
        NN_labels, NN_dists = One_NN_search(qs, holdout_dataset, i, p)
        for j in range(p):
            # NN_true and NN_false are NN_dists for true positives and false positives in j_th class
            NN_true, NN_false, precision = caculate_precision(NN_labels, NN_dists, qs, j)
            DN_matrix[i,j] = [NN_true, NN_false]
            C_matrix[i,j] = precision
            
    return C_matrix, DN_matrix

def classify():
    """ Classify a test set
    """

def prepare_ACS_dataset(dataset, dumpfile):
    """ 转换数据格式，并进行永久化存储
    dataset: 之前的存储方式
    dumpfile: 目标存储文件
    """
    ACS_dataset = []
    for topic_id, ins_feature, true_level in dataset:
        target_comment_count = ins_feature[0][0]
        prediction_date_point = ins_feature[0][1]
        target_date_point = ins_feature[0][2]
        topic_id = ins_feature[0][3]
        prediction_comment_count = ins_feature[0][4]
        
        num_feature = len(ins_feature[1])
        ts_length = len(ins_feature) - 1 # 时间序列的长度
        
        feature = np.zeros((num_feature, ts_length), float)
        for i in range(ts_length):
            feature[:, i] = ins_feature[i+1]
        
        ins = Instance(topic_id, prediction_date_point, target_date_point, \
            prediction_comment_count, target_comment_count, feature, true_level)
        ACS_dataset.append(ins)
            
    pickle.dump(ACS_dataset, dumpfile)
