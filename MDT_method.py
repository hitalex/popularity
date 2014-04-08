#coding=utf8

"""
实现The ICDM 2013 method: Adjusted Confidence Score

Notes:
1, 这里将对之前的dataset进行转换，使用新的数据存储方式
2，本方法中要求计算1NN，但是有时会存在多个最近邻，所以这里打算先进行投票，然后最近距离采用均值
"""
import pickle
import math

import numpy as np
import scipy.stats

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
    count = int(len(holdout_dataset) * percent)
    return holdout_dataset[:count]
    
def majority_vote(NN_list, p):
    """ majority vote of each class
    """
    votes = [0] * p
    count = len(NN_list)
    for i in range(count):
        level = NN_list[i].true_level
        votes[level] += 1
        
    return votes.index(max(votes))

def prob_density_estimation(NN_dists, target_dist, h):
    """ Use the non-parametric parzen window to evaluate probilities density
    """
    n = len(NN_dists)
    if n == 0:
        return 1e-6
        
    hn = h / math.sqrt(n)
    s = 0
    gaussian_norm = scipy.stats.norm(0, 1)
    for i in range(n):
        x = (target_dist - NN_dists[i]) / hn
        s += (1.0/hn * gaussian_norm.pdf(x))
        
    s /= n
    
    return s
    
def validate_distance_based_classification(DN_matrix):
    """ 验证distance based measurement是否能够区分true positive和false positive
    """
    import ipdb; ipdb.set_trace()
    import matplotlib.pyplot as plt
    m, p = DN_matrix.shape
    for i in range(m):
        for j in range(p):
            NN_true = DN_matrix[i,j][0]
            NN_false = DN_matrix[i,j][1]
            plt.figure()
            n, bins, patches = plt.hist(NN_true, bins=10, normed=0, facecolor='green', alpha=0.5, hold=True)
            n, bins, patches = plt.hist(NN_false, bins=10, normed=0, facecolor='red', alpha=0.5, hold=True)
            plt.title('Dimension: %d, Class index: %d' % (i, j))
            plt.show()
    
def One_NN_search(ins, holdout_dataset, dim_index, p):
    """ One nearest neighbour search in dataset
    ins: the query instance
    holdout: holdout dataset
    dim_index: which dim is to search
    q: number of classes

    Note: 因为可能会找出多个最近邻，所以在这里会先进行投票，假如得票最多的是第i类，那么
    dist的值则由第i类中所有的近邻的dist进行平均得到
    """
    total = len(holdout_dataset)
    min_dist = float('inf')
    NN_list = []
    for j in range(total):
        if ins.topic_id == holdout_dataset[j].topic_id:
            continue
        prediction_date_point = ins.prediction_date_point
        vec1 = ins.feature[dim_index, :prediction_date_point+1]
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
        NN_ins = NN_list[0]
        return NN_ins.true_level, min_dist
    else:
        label = majority_vote(NN_list, p)
        return label, min_dist
        
    return -1, -1
    
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
        true_label = qs[i].true_level
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
    #import ipdb; ipdb.set_trace()
    qs = get_query_samples(holdout_dataset, 0.8)
    query_count = len(qs)
    
    C_matrix = np.zeros((m, p), dtype=float)
    DN_matrix = np.zeros((m, p), dtype=object)
    
    for i in range(m):
        print 'Caculating confidence score for dimension:', i
        NN_labels = [-1] * query_count
        NN_dists = [-1] * query_count
        for j in range(query_count):            
            NN_labels[j], NN_dists[j] = One_NN_search(qs[j], holdout_dataset, i, p)
            
        for j in range(p):
            # NN_true and NN_false are NN_dists for true positives and false positives in j_th class
            NN_true, NN_false, precision = caculate_precision(NN_labels, NN_dists, qs, j)
            
            if len(NN_true) == 0:
                print 'Warning: NN_true is empty for dim: %d, class: %d' % (i, p)
            if len(NN_false) == 0:
                print 'Warning: NN_false is empty for dim: %d, class: %d' % (i, p)
                
            DN_matrix[i,j] = [NN_true, NN_false]
            C_matrix[i,j] = precision
            
    return C_matrix, DN_matrix
    
def caculate_adC(pl, target_dis, dim_index, DN_matrix, C_matrix, h):
    """ 
    pl: the predicted label
    target_dis: the current dis
    dim_index: which dimension we are in
    DN_matrix and C_matrix: 
    """
    #import ipdb; ipdb.set_trace()
    acc = C_matrix[dim_index, pl] # P(pl=tl)
    
    from scipy.stats import kde # use gaussian kernel density estimation
    density = kde.gaussian_kde(DN_matrix[dim_index, pl][0])
    tmp1 = density(target_dis) # P(dis|pl=tl)
    density = kde.gaussian_kde(DN_matrix[dim_index, pl][1])
    tmp2 = density(target_dis) # P(dis|pl!=tl)

    prob = tmp1 * acc / (tmp1 * acc + tmp2 * (1-acc))
    
    return prob
    
def classify(test_dataset, train_dataset, C_matrix, DN_matrix, p, m):
    """ Classify instances in testset
    """
    #import ipdb; ipdb.set_trace()
    assert(p == 2) # 这里只适用2两类的情况
    parzen_window_width = [1] * m
    total = len(test_dataset)
    index = 0
    pred_labels = [-1] * total
    true_labels = [-1] * total
    for test_ins in test_dataset:
        true_labels[index] = test_ins.true_level
        adC_vector = np.array([0] * m, float)
        NN_labels = [-1] * m
        for i in range(m):
            NN_labels[i], NN_dist = One_NN_search(test_ins, train_dataset, i, p)
            adC_vector[i] = caculate_adC(NN_labels[i], NN_dist, i, DN_matrix, C_matrix, parzen_window_width[i])
        
        score = np.array([0] * p, float)
        for i in range(m):
            pl = NN_labels[i]
            score[pl] += adC_vector[i]
            score[1-pl] += (1-adC_vector[i])
        
        score = score / np.sum(score)
        index_max = np.argmax(score)
        pred_labels[index] = index_max
        
        print 'Confidence score:', score
        print 'True label: %d, prediction label: %d' % (true_labels[index], pred_labels[index])
        #index_max = np.argmax(adC_vector)
        #pred_labels[index] = NN_labels[index_max]
        index += 1
        
    return pred_labels, true_labels

def prepare_MDT_dataset(dataset, dumpfile):
    """ 转换数据格式，并进行永久化存储
    dataset: 之前的存储方式
    dumpfile: 目标存储文件
    """
    #import ipdb; ipdb.set_trace()
    MDT_dataset = []
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
        MDT_dataset.append(ins)
            
    f = open(dumpfile, 'w')
    pickle.dump(MDT_dataset, f)
    f.close()
    
if __name__ == '__main__':
    print 'Loading the train and test data...'
    f = open('pickle/MDT_train.pickle', 'r')
    train_dataset = pickle.load(f)
    f.close()
    f = open('pickle/MDT_test.pickle', 'r')
    test_dataset = pickle.load(f)
    f.close()
    
    print 'Caculating the C_matrix and DN_matrix...'
    p = 2
    m = (train_dataset[0].feature.shape)[0]
    print 'Number of classes: %d, Number of dimensions: %d' % (p, m)
    C_matrix, DN_matrix = caculate_confidence_score(train_dataset, p, m)
    print 'C_matrix:', C_matrix
    #validate_distance_based_classification(DN_matrix)
    
    print 'Classifying test intances...'
    pred_labels, true_labels = classify(test_dataset, train_dataset, C_matrix, DN_matrix, p, m)
    import sklearn.metrics 
    print sklearn.metrics.classification_report(true_labels, pred_labels)
    print 'Acc score:', sklearn.metrics.accuracy_score(true_labels, pred_labels)
