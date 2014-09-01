#coding=utf8

# cross validation for k

from popularity import *
from utils import load_id_list, transform_ts, down_sampling_dataset
from sklearn.cross_validation import KFold

"""
交叉验证：3-fold交叉验证
"""

def make_cv_dataset(dataset, index_list):
    cv_dataset = [0] * len(index_list)
    i = 0
    for index in index_list:
        cv_dataset[i] = dataset[index]
        i += 1
        
    return cv_dataset

def select_k(group_id, topic_list, percentage_threshold, prediction_date, response_time, cvk):
    # sampling interval
    gaptime = timedelta(hours=5)
    target_date = prediction_date + response_time
    num_feature = int(prediction_date.total_seconds() / gaptime.total_seconds())
    print 'Number of features: ', num_feature
    
    #percentage_threshold = 0.7
    alpha = 1/percentage_threshold
    pop_level = [25, 50, float('inf')]  # group: zhuangb
    
    print 'Generating training and test dataset...'
    dataset, comment_count_dataset, Bao_dataset, category_count_list = prepare_dataset(group_id, \
        topic_list, gaptime, pop_level, prediction_date, target_date, alpha, percentage_threshold)
        
    print 'Down-sampling the datasets...'
    dataset, comment_count_dataset, Bao_dataset, category_count_list = down_sampling_dataset(dataset, \
        comment_count_dataset, Bao_dataset, category_count_list)
        
    total = len(dataset)
    n_folds = 3
    kf = KFold(total, n_folds)
    IPW_acc_list = []
    for cv_train_index, cv_test_index in kf:
        train_set = make_cv_dataset(dataset, cv_train_index)
        test_set = make_cv_dataset(dataset,cv_test_index)
        train_cnt = len(train_set)
        print 'Training: %d, Test: %d' % (train_cnt, total-train_cnt)
        print 'Category 0: %d, Category 1: %d ' % (category_count_list[0] , category_count_list[1])
        print 'Imbalance ratio: ', category_count_list[0] * 1.0 / category_count_list[1]
        
        num_level = 2
        num_factor = len(train_set[0][1][1])
        
        print 'The proposed model:'
        print 'Caculating instance prior score...'
        prior_score = -1
        mutual_knn_graph_list = None
        #prior_score = caculate_instance_prior_confidence_score(train_set, k, num_level = 2) # for instance_prior_weighting3.py
        topic_popularity, prior_score, mutual_knn_graph_list = caculate_instance_prior_confidence_score(train_set, test_set, cvk, num_factor, num_level = 2) # for IPW_mutual_knn.py

        print 'Classify test instances...'
        y_true, y_pred, comment_true, comment_pred, give_up_list, prediction_list, factor_prediction = \
            classify(train_set, test_set, cvk, num_factor, num_level, prior_score, topic_popularity, mutual_knn_graph_list)
        # evaluate results
        print 'Number of give-ups: ', len(give_up_list)
        IPW_acc = classification_evaluation(y_true, y_pred)
        
        IPW_acc_list.append(IPW_acc)
        
    return IPW_acc_list

if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    topiclist_path = 'data-dynamic/TopicList-' + group_id + '-filtered.txt' # for douban-group
    #topiclist_path = 'data-dynamic/' + group_id + '-post-list.txt' # for Tianya dataset
    
    print 'Reading topic list from file:', topiclist_path
    topic_list = load_id_list(topiclist_path)
    print 'Number of total topics loaded: ', len(topic_list)
    
    # for test
    #topic_list = topic_list[:50]
    
    threshold_p = 0.7
    prediction_date_tr = timedelta(hours=50)
    response_time_delta = timedelta(hours=25)
    #cvk_list = [1, 3, 5, 7]
    cvk_list = [9]
    for cvk in cvk_list:
        print 'CV for k: ', cvk
        IPW_acc_list = select_k(group_id, topic_list, threshold_p, prediction_date_tr, response_time_delta, cvk)
        print IPW_acc_list
        print 'avg_IPW_acc: ', sum(IPW_acc_list)*1.0/len(IPW_acc_list)

