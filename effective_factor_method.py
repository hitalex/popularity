#coding=utf8

"""
目前的想法：非参数化的方法
首先，对于训练集中的所有帖子，分别在不同的动态因素（假设有m个动态因素）下查找其最近邻，那么
就有m个结果。对照这m个结果和真实标签的对应关系，记录下是哪些动态因素使得分类正确。
在测试时，对于一个测试数据t，分别在不同的动态因素下寻找k近邻：对于每个动态因素dy，其最近邻假设为{b1, b2,...bk}，
那么如果bi的确可以在dy下分类正确，那么就接受bi的标签为候选标签。
"""
import operator

from score_ranking import get_instance_distance

def trusted_vote(nn_level_list, num_level=2, majority_threshold = 0.8):
    """ 直接找到满足要求的level，否则返回-1
    majority_threshold: 其中一个level的比例大于此值，则值得相信
    """
    level_count = [0] * num_level
    for level in nn_level_list:
        level_count[level] += 1
        
    total = len(nn_level_list)
    for i in range(num_level):
        level_count[i] = level_count[i] * 1.0 / total
        if level_count[i] >= majority_threshold:
            return i
            
    return -1

def get_knn_level_list(distance_list, k):
    """ 获取k近邻的level标签
    按照如下方式：首先获取最近邻，如果最近邻数超过k个，则返回全部的最近邻；
    如果最近邻数不足k个（可能有相同距离），则考虑次近邻；
    """
    knn_dis = [0] * k
    knn_dis[0] = distance_list[0][1]
    knn_dis_count = [0] * k
    current_k_index = 0
    total = len(distance_list)
    nn_level_list = []
    for i in range(1, total):
        dis = distance_list[i][1]
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
    for i in range(total):
        knn_level_list[i] = distance_list[i][2]
        if isinstance(knn_level_list[i], list):
            import ipdb; ipdb.set_trace()
        
    return knn_level_list, distance_list[:total]

def effective_factor_knn(train_set, test_ins, k, effective_factors):
    """ 给定有效因素进行分类
    """
    train_count = len(train_set)
    num_factor = len(test_ins[1]) # get the number of features
    test_topic_id = test_ins[0][3]
    # 初始化所有的topic的rank score

    topic_popularity = dict()    # topic_id ==> (level, comment_count)
    for train_topic_id, train_ins, level in train_set:
        target_comment_count = train_ins[0][0]
        prediction_comment_count = train_ins[0][4]
        ratio = target_comment_count * 1.0 / prediction_comment_count
        topic_popularity[train_topic_id] = (level, target_comment_count, prediction_comment_count, ratio)
        
    topic_score_list = [] # 每个feature都保存一份score list
    """
    注：分别在不同的dynamic factor中查找最近邻，然后将这些最近邻组合起来投票
    """
    #import ipdb; ipdb.set_trace()
    pred_level_list = []
    for findex in range(num_factor):
        #print 'Caculating score and rank for feature: ', findex
        distance_list = [0] * train_count
        index = 0
        #import ipdb, ipdb.set_trace()
        for train_topic_id, train_ins, level in train_set:
            if not findex in effective_factors[train_topic_id]:
                continue
            dis = get_instance_distance(test_ins, train_ins, findex)
            distance_list[index] = [train_topic_id, dis, level]
            index += 1
        
        distance_list = distance_list[:index]
        # 按照dis进行升序排序
        distance_list.sort(key=operator.itemgetter(1), reverse=False)
        level_list, nn_list = get_knn_level_list(distance_list, k)
        
        pred_level = trusted_vote(level_list, num_level=2, majority_threshold = 0.66)
        if pred_level != -1:
            pred_level_list.append(pred_level)

    #print '\nOverall pred level list: ', pred_level_list
    
    if len(pred_level_list) == 0:
        return [], '', 0
    
    return pred_level_list, '', 0
    
    num_neighbour = len(knn_list)
    knn_level = [0] * num_neighbour
    knn_topic_id = [0] * num_neighbour # 真正的k个近邻的topic id
    # 将k近邻的评论数进行加权平均，将score值作为权值
    weighted_num_comment = 0
    weighted_ratio = 0
    total_score = 0
    for i in range(num_neighbour):
        topic_id                = knn_list[i][0]
        dis                     = knn_list[i][1]
        prediction_comment_count= knn_list[i][2]
        target_comment_count    = knn_list[i][3]        
        level                   = knn_list[i][4]
        ratio                   = knn_list[i][5]
        
        knn_level[i] = level
        knn_topic_id[i] = topic_id
        
        score = 1
        total_score += score
        weighted_num_comment += (score * target_comment_count)
        weighted_ratio += (score * ratio)
    
    weighted_num_comment = weighted_num_comment * 1.0 / total_score
    weighted_ratio = weighted_ratio * 1.0 / total_score
    weighted_ratio_pred = test_ins[0][4] * weighted_ratio
    
    return knn_level, knn_topic_id, weighted_ratio_pred


def find_effective_factor(train_set, k):
    """ 找出每个训练样本在哪些dynamic factor下能够分类成功
    """
    #import ipdb; ipdb.set_trace()
    num_factor = len(train_set[0][1][1]) # get the number of dynamic factors
    train_count = len(train_set)
    
    effective_factors = dict() # topic_id ==> effective factor set
    for topic_id, ins, true_level in train_set:
        effective_factor_set = set()
        for findex in range(num_factor):
            distance_list = [0] * (train_count-1)
            index = 0
            for train_topic_id, train_ins, train_true_level in train_set:
                if train_topic_id == topic_id: # 不考虑自身
                    continue
                dis = get_instance_distance(ins, train_ins, findex)
                distance_list[index] = [train_topic_id, dis, train_true_level]
                index += 1
            
            distance_list = distance_list[:index]
            distance_list.sort(key=operator.itemgetter(1), reverse=False)
            level_list, nn_list = get_knn_level_list(distance_list, k)
            pred_level = trusted_vote(level_list, num_level=2, majority_threshold = 0.66)
            if pred_level == true_level:
                effective_factor_set.add(findex)
        
        print 'Effecitive factors for %s: %r' % (topic_id, effective_factor_set)
        effective_factors[topic_id] = effective_factor_set
        
    return effective_factors
            
