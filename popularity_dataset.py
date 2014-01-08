#coding=utf8

"""
只采用dynamic factor作为特征，使用instance-based方法进行预测
"""

import codecs
from datetime import datetime, timedelta
import math
import os.path
from random import choice, shuffle

from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np

from utils import load_id_list, transform_ts, down_sampling_dataset
#from score_ranking import score_ranking_knn, get_instance_distance
from score_ranking_vote import score_ranking_knn, get_instance_distance, caculate_class_prior_confidence_score
from effective_factor_method import effective_factor_knn, find_effective_factor
from baseline_methods import SH_model, ML_model, MLR_model, knn_method, ARIMA_model, Bao_method
from plot.factor_relevance_plot import *

# 评论数量的阈值，如果评论数量小于该值，则不考虑
MIN_COMMENT = 5
# 评论数量的最大值
MAX_COMMENT = 1000
# 可以看作是viral的最少评论数, 目前不打算在此进行约束
# 目标时刻的评论数用threshold来约束
VIRAL_MIN_COMMENT = 50
# 抓取内容的时间。如果target_date在此之后，则同样不进行预测
DEADLINE = datetime(2013, 11, 15)
# 在开始预测时，最少需要拥有的comment数量
MIN_COMMENT_PREDICTION_DATE = 10

def get_level_index(num_comment, pop_level):
    """ Get the populairty leve according to the number of comments
    """
    index = 0
    while num_comment > pop_level[index]:
        index += 1

    return index
    
def get_topic_category(thread_pubdate, comment_feature_list, threshold):
    """ 根据帖子在特定一段时间（例如一天）内的获得的comment的峰值所占总评论数的比例
    来对帖子进行分类
    See WSDM'11, The tube over time...
    """
    assert(threshold > 0 and threshold < 1)
    interval = timedelta(hours=12) # 此时间间隔可调
    # 第一步需要找到，从帖子发布开始，所有interval的comment数量的峰值
    peak_comment = 0
    total_comment = len(comment_feature_list)
    pre_check_date = thread_pubdate
    pre_index = 0
    next_check_date = pre_check_date + interval
    index = 0
    while index < total_comment:
        # 搜索在next_check_date之前，但离的最近的comment
        pubdate, feature = comment_feature_list[index]
        if pubdate < next_check_date:
            index += 1
            continue
        else:
            current_interval_comment = index - pre_index
            if current_interval_comment > peak_comment:
                peak_comment = current_interval_comment

            pre_check_date = pubdate
            next_check_date = pre_check_date + interval
            pre_index = index
            
    current_interval_comment = index - pre_index
    if current_interval_comment > peak_comment:
        peak_comment = current_interval_comment
    
    percentage_peak = peak_comment * 1.0 / total_comment
    
    cat = 0
    
    """
    if percentage_peak <= threshold:
        cat = 0 # viral
    elif percentage_peak > threshold and percentage_peak <= 1 - threshold:
        cat = 1 # quality
    else:
        cat = 2 # junk
    """
    if percentage_peak <= threshold and total_comment > VIRAL_MIN_COMMENT:
        cat = 1 # viral
    else:
        cat = 0
           
    return cat
    
def get_comment_percentage_category(total_comment, prediction_comment_count, percentage_threshold = 0.6):
    """ 分类标准：某个帖子在prediction_date_point时的comment数是否已经占所有comment总数的percentage_threshold
    """
    if total_comment > VIRAL_MIN_COMMENT and prediction_comment_count * 1.0 / total_comment <= percentage_threshold:
        cat = 1
    else:
        cat = 0
        
    return cat
    
def genereate_topic_feature(comment_feature_list, thread_pubdate, gaptime):
    """ 将comment_feature_list转换为interval_list
    """
    num_comment = len(comment_feature_list)
    comment_index = 0
    topic_feature = []
    flag = True
    curr_date = thread_pubdate
    while True:
        curr_date += gaptime
        while True:
            pubdate = comment_feature_list[comment_index][0]
            feature = comment_feature_list[comment_index][1]
            if pubdate < curr_date:
                comment_index += 1
            else:
                break
            if comment_index >= num_comment:
                flag = False
                break
        
        feature = comment_feature_list[comment_index-1][1]
        topic_feature.append(feature)
        if not flag:
            break
    
    return topic_feature
    
def transform_count_feature(topic_feature, factor_index_list):
    """ 按照NIPS'13文章中的方法，将feature vector进行转化
    factor_index_list: 需要进行归一化的factor index列表
    注意：NIPS'13文章中的方法只适用与count相关的feature
    """
    num_feature = len(topic_feature)
    p = [0] * num_feature
    p = np.array(p, float)
    for i in factor_index_list:
        for j in range(num_feature):
            p[j] = topic_feature[j][i]
            
        p = transform_ts(p)
        
        for j in range(num_feature):
            topic_feature[j][i] = p[j]
            
    return topic_feature
        
def prepare_dataset(group_id, topic_list, gaptime, pop_level, prediction_date, target_date, alpha, percentage_threshold):
    """ Load tranining or test dataset for each topic from dynamic featur data   
    pop_level: popularity level for the whole dataset
    num_feature: 这里实际上是考察多少个comment，并不是完全按照时间来预测
    target_date: 在帖子发布target_date日期之后开始预测
    alpha: the ratio of target date comment count and prediction date comment cout
    Note: 在准备数据集的同时，需要记录从哪一个interval开始预测
    """
    assert(alpha >= 1)
    print 'Prepare dataset for group: ', group_id
    dataset = []
    comment_count_dataset = []
    # dataset for Bao's method: prediction_comment_count, link density, diffusion depth
    Bao_dataset = []    
    prediction_date_point = int(prediction_date.total_seconds() / gaptime.total_seconds())
    category_count_list = [0, 0, 0]
    level_count_list = [0] * len(pop_level)
    for topic_id in topic_list:
        path = 'data-dynamic/' + group_id + '/' + topic_id + '.txt'
        if not os.path.exists(path):
            continue
        f = open(path, 'r')
        # read the first line
        line = f.readline().strip()
        
        try:
            feature_dict = eval(line)
        except SyntaxError:
            print 'Format error in topic:', path
            continue
        
        publisher = feature_dict['lz']
        thread_pubdate = datetime.strptime(feature_dict['pubdate'], '%Y-%m-%d %H:%M:%S')
        total_comment = feature_dict['num_comment']
        
        if total_comment < MIN_COMMENT or DEADLINE < thread_pubdate + target_date:
            continue
        
        current_comment_count = 0
        prediction_date_flag = False
        target_date_flag = False      # 标记用于检测目标date的interval的index
        target_comment_count = -1     # 目标日期节点时的comment数量
        prediction_comment_count = -1
        comment_feature_list = []   # 按照评论来收集feature
        comment_count_list = []     # 用于ML模型，只用来收集comment count特征
        for line in f:
            line = line.strip()
            current_comment_count += 1
            # eval the str as feature dict
            nan = float('nan')
            feature_dict = eval(line)
            # basic info
            cid                     = feature_dict['cid']
            pid                     = feature_dict['pid']
            pubdate                 = datetime.strptime(feature_dict['pubdate'], '%Y-%m-%d %H:%M:%S')
            
            # author-reply features
            mean_degree             = feature_dict['mean_degree']
            clustering_coefficient  = feature_dict['clustering_coefficient']
            reply_density           = feature_dict['reply_density']
            num_authors             = feature_dict['num_authors']
            
            # comment tree features
            diffusion_depth         = feature_dict['diffusion_depth']
            avg_weighted_depth_sum  = feature_dict['avg_weighted_depth_sum']
            tree_link_density       = feature_dict['tree_density']
            #author_reply_max_cohesions = feature_dict['author_reply_max_cohesions']
            
            # comment_author two-mode network
            ca_mean_degree              = feature_dict['ca_mean_degree']
            #ca_avg_local_transitivity   = feature_dict['ca_avg_local_transitivity']
            #ca_reply_density            = feature_dict['ca_reply_density']
            #ca_clustering_coefficient   = feature_dict['ca_clustering_coefficient']
            #ca_max_cohesions        = feature_dict['ca_max_cohesions']
            
            delta_comment_count = 0 # 相较于上个interval增加的comment
            # features: [comment count, mean degree]
            #feature = [current_comment_count, tree_link_density, mean_degree]
            # 当只取current_comment_count作为feature时，相当于简单knn算法
            feature = [diffusion_depth]
            
            comment_feature_list.append((pubdate, feature))
            # comment_count_list只记录了当前的评论数信息，用于baseline方法计算
            comment_count_list.append((pubdate, current_comment_count))
            
            if not prediction_date_flag and pubdate >= thread_pubdate + prediction_date:
                prediction_comment_count = current_comment_count
                # for Bao's method
                prediction_link_density = tree_link_density
                prediction_diffusion_depth = diffusion_depth
                prediction_date_flag = True
            
            if pubdate >= thread_pubdate + target_date:
                target_date_flag = True
                target_comment_count = current_comment_count
            
            # 最多只考虑 MAX_COMMENT 个评论
            if len(comment_feature_list) > MAX_COMMENT:
                break
                
        # 如果在预测时的comment数量小于MIN_COMMENT_PREDICTION_DATE，则放弃预测
        if prediction_comment_count < MIN_COMMENT_PREDICTION_DATE:
            continue
            
        # 如果最后一个评论的时间还不到target date，则不考虑这些帖子
        if not target_date_flag:
            continue
            
        # 接下来将comment_feature_list转换为topic_feature
        topic_feature = genereate_topic_feature(comment_feature_list, thread_pubdate, gaptime)
        comment_count_feature = genereate_topic_feature(comment_count_list, thread_pubdate, gaptime)
        # transform with delta features
        topic_feature = transform_count_feature(topic_feature, factor_index_list = [])
        # 获得topic的category
        cat = get_topic_category(thread_pubdate, comment_feature_list, percentage_threshold)
        #cat = get_comment_percentage_category(target_comment_count, prediction_comment_count, percentage_threshold)
        #cat = get_comment_percentage_category(total_comment, prediction_comment_count, percentage_threshold)
        
        category_count_list[cat] += 1
        # first feature vector，记录其他信息
        topic_feature.insert(0, [0, 0, 0, 0, 0])
        comment_count_feature.insert(0, [0, 0, 0, 0, 0])
        # 这里只记录在target_date_point时间点的评论数量
        comment_count_feature[0][0] = topic_feature[0][0] = target_comment_count
        comment_count_feature[0][1] = topic_feature[0][1] = prediction_date_point
        # 设置为最后一个interval
        target_date_point = int(target_date.total_seconds() / gaptime.total_seconds())
        if target_date_point > len(topic_feature):
            target_date_point = len(topic_feature)
        comment_count_feature[0][2] = topic_feature[0][2] = target_date_point
        comment_count_feature[0][3] = topic_feature[0][3] = topic_id
        comment_count_feature[0][4] = topic_feature[0][4] = prediction_comment_count
        
        # the data and the label
        #dataset.append((topic_id, topic_feature, popularity_level))
        dataset.append((topic_id, topic_feature, cat))
        comment_count_dataset.append((topic_id, comment_count_feature, cat))
        
        Bao_dataset.append((topic_id, prediction_comment_count, target_comment_count, \
            prediction_link_density, prediction_diffusion_depth, cat))
    
    print '过滤后最终数据集中的topic总数：', len(dataset)
    print 'Category distribution: ', category_count_list
    print 'Imbalance ratio: ', category_count_list[0] * 1.0 / category_count_list[1]
    #print 'Popularity level distribution:', level_count_list
    
    return dataset, comment_count_dataset, Bao_dataset, category_count_list
    
def save_filtered_topics(group_id, dataset):
    """
    功能：保存那些经过筛选的topic id
    设置综合性的满足要求的所有数据集，groupid为:kong
    """
    #import ipdb; ipdb.set_trace()
    import os
    base_path = '/home/kqc/Dropbox/projects/popularity/'
    dir_path = 'data-dynamic/kong2/'
    
    path = 'data-dynamic/TopicList-' + group_id + '-filtered.txt'
    f = open(path, 'w')
    path = 'data-dynamic/TopicList-kong2-filtered.txt'
    ft = open(path, 'a')
    for topic_id, topic_feature, popularity_level in dataset:
        f.write(topic_id + '\n')
        ft.write(topic_id + '\n')
        
        spath = base_path + 'data-dynamic/' + group_id + '/' + topic_id + '.txt'
        tpath = base_path + 'data-dynamic/' + 'kong2' + '/' + topic_id + '.txt'
        args = '/bin/cp ' + spath + ' ' + tpath
        # 将topic信息文件复制到目标文件
        os.popen(args)
    
    f.close()
    ft.close()
    
def main(group_id):

    topiclist_path = 'data-dynamic/TopicList-' + group_id + '-shuffled.txt'
    topic_list = load_id_list(topiclist_path)
    print 'Number of total topics loaded: ', len(topic_list)
    pop_level = [25, 50, float('inf')]  # group: zhuangb
    
    # prediction_date 的含义为：在帖子发布 prediction_date 时间后，开始预测
    # target_date 的含义为：预测在 target_date 处的评论数量
    # 以上两个参数可以调节
    # 设置采样的间隔
    gaptime = timedelta(hours=5)
    prediction_date = timedelta(hours=10*5)
    response_time = timedelta(hours=50)
    target_date = prediction_date + response_time
    
    # 计算每个topic在prediction_date前会有多少个interval
    num_feature = int(prediction_date.total_seconds() / gaptime.total_seconds())
    print 'Number of features: ', num_feature
    
    alpha = 1.5
    percentage_threshold = 0.5
    print 'Generating training and test dataset...'
    dataset, comment_count_dataset, Bao_dataset, category_count_list = prepare_dataset(group_id, \
        topic_list, gaptime, pop_level, prediction_date, target_date, alpha, percentage_threshold)
    # 保存那些经过筛选的topic id
    #save_filtered_topics(group_id, dataset)
    print 'Ploting factor propagation'
    #factor_propagation_plot(dataset, num_feature)
    #topic_propagation_plot(dataset, num_feature)
    #return 
    
    # 调整所有帖子的顺序
    # 在调试阶段，暂且不shuffle dataset，避免每次结果都不一样
    #shuffle(dataset)
    
    # 注意：每次使用的数据集是不同的
    total = len(dataset)
    train_cnt = total * 4 / 5
    train_set = dataset[:train_cnt]
    test_set = dataset[train_cnt:]
    print 'Training: %d, Test: %d' % (train_cnt, total-train_cnt)
    print 'Category 0: %d, Category 1: %d ' % (category_count_list[0] , category_count_list[1])
    print 'Imbalance ratio: ', category_count_list[0] * 1.0 / category_count_list[1]
    
    dataset, comment_count_dataset, Bao_dataset, category_count_list = down_sampling_dataset(dataset, \
        comment_count_dataset, Bao_dataset, category_count_list)    
    print 'After down sampling...'
    total = len(dataset)
    train_cnt = total * 4 / 5
    train_set = dataset[:train_cnt]
    test_set = dataset[train_cnt:]
    
    print 'Training: %d, Test: %d' % (train_cnt, total-train_cnt)
    print 'Category 0: %d, Category 1: %d ' % (category_count_list[0] , category_count_list[1])
    print 'Imbalance ratio: ', category_count_list[0] * 1.0 / category_count_list[1]

    
if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    main(group_id)

