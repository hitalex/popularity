#encoding=utf8

def seg_chinese(chinese_str):
    """中文分词
    Note: 目前采用jieba分词. jieba项目主页：https://github.com/fxsjy/jieba
    """
    import jieba
    seg_list = jieba.cut(chinese_str)
    return " ".join(seg_list)
    
def my_min_max_scaler(data, a=0, b=1.0):
    """ 将a中的数据归一化到min_value和max_value之间
    a and b: 指定的区间
    """
    assert(b > a)
    
    count = len(data)
    min_value = min(data)
    max_value = max(data)
    
    if min_value == max_value:
        return [0] * count
        
    for i in range(count):
        data[i] = (b-a) * (data[i] - min_value) * 1.0 / (max_value - min_value)
        
    return data
            
def down_sampling_dataset(dataset, comment_count_dataset, Bao_dataset, category_count_list):
    """ 下采样技术：对有大多数样本的类进行下采样，使得两类的样本数相差不多
    Note: 这里需要保证所有的方法使用的数据集相同，包括baseline方法和当前提出的方法
    """
    from random import random, shuffle
    ratio = category_count_list[0] * 1.0 / category_count_list[1]
    if ratio < 1:
        majority_category = 1
    elif ratio > 1:
        ratio = 1 / ratio
        majority_category = 0
    else:
        return dataset, comment_count_dataset, Bao_dataset, category_count_list
    
    selected_index = []
    new_category_count_list = [0, 0]
    print 'Downsampling category: ', majority_category
    total = len(dataset)
    # 在测试期间，需要保证训练数据的一致性，所以暂时改变down sampling方法
    majortity_count = 0
    minority_class_count = min(category_count_list[:2])
    
    print 'Minority class count:', minority_class_count
    #import ipdb; ipdb.set_trace()
    for i in range(total):
        # TODO: 需要说明的是类别标签信息必须在最后一个item上，否则会出错
        cat = dataset[i][-1]
        assert(cat == 0 or cat == 1)
        #if cat == majority_category: # 只对majortiy class进行处理
        #    if random() > ratio: # 未被选中
        #        continue
        
        if cat == majority_category:
            majortity_count += 1
            if majortity_count > minority_class_count:
                continue
        
        selected_index.append(i)
        #print 'Select index: %d, cat is %d.' % (i, cat)
        
        new_category_count_list[cat] += 1
    
    shuffle(selected_index)
    """
    for i in selected_index:
        cat = dataset[i][-1]
        print 'Select index: %d, cat is %d.' % (i, cat)
    """
    dataset = [dataset[i] for i in selected_index]
    comment_count_dataset = [comment_count_dataset[i] for i in selected_index]
    Bao_dataset = [Bao_dataset[i] for i in selected_index]
        
    return dataset, comment_count_dataset, Bao_dataset, new_category_count_list
    
def ratio_accuracy_distribution_plot(y_true, y_pred, test_set, group_id, factor_name):
    """
    查看对于不同的dynamic factor，它们在不同的ratio上的预测结果
    """
    ratio_list = []
    index = 0
    #import ipdb; ipdb.set_trace()
    for test_topic_id, test_ins, true_level in test_set:
        if y_pred[index][0] == y_true[index]:
            target_comment_cnt = test_ins[0][0]
            prediction_comment_cnt = test_ins[0][4]
            ratio = target_comment_cnt * 1.0 / prediction_comment_cnt
            if ratio > 20:
                continue
            ratio_list.append(ratio)
        index += 1
        
    import matplotlib.pyplot as plt
    import numpy as np
    xmin = int(min(ratio_list))
    xmax = int(max(ratio_list)) + 1
    
    plt.title(group_id + ': ' + factor_name)
    bin_range = np.array(range(1*10, 20*10, 1)) * 1.0 / 10
    n, bins, patches = plt.hist(ratio_list, bins=bin_range, normed=0, facecolor='green', alpha=0.5)
    plt.show()
    
def transform_ts(sp, alpha = 1.2, T_smooth = 3):
    """ 按照NIPS'13文章中的预处理方法处理单个ts
    alpha: how to emphasize the large spikes
    T_smooth: the smoothing parameter
    """
    #import ipdb; ipdb.set_trace()
    
    import numpy as np
    import math
    count = len(sp)
    # extend the array
    p = np.array(sp, float)
    p = np.insert(p, 0, 0)
    
    tmp = [p[i] - p[i-1] for i in range(1, count+1)]
    p = tmp
    
    """
    #maxp = max(p)
    maxp = sum(p) / count
    # 如果为0，则不转换
    if maxp == 0:
        maxp = 1
    # normalize p
    p = p * 1.0 / maxp
    """
    
    tmp = np.cumsum(p)
    p = [p[i]/tmp[i] for i in range(count)]
    p = np.array(p, float)
    
    p = np.insert(p, 0, 0)
    # get the delta feature
    p1 = np.array(p, float)
    p2 = np.array(p, float)
    for i in range(1, count+1):
        # emphasize spikes
        p1[i] = (abs(p[i] - p[i-1])) ** alpha
        
        # smoothe the sequence
        start_index = i - T_smooth + 1
        if start_index < 0:
            start_index = 0
        p2[i] = sum(p1[start_index:i+1])
        
        # in case it will be zero
        if p2[i] == 0:
            p2[i] = 1e-06
            
        # get the log transformed
        p2[i] = math.log(p2[i])
        
    return p2[1:]
    
def is_between(now, start_date, end_date):
    """ 判断给定的时间是否在起止时间内
    """
    from datetime import datetime
    
    if now >= start_date and now < end_date:
        return True
    else:
        return False
        
def load_id_list(file_path):
    """ 从文件内导入所有的id，每行一个，返回这些id的list
    """
    f = open(file_path, 'r')
    id_list = []
    for line in f:
        line = line.strip()
        if line != '':
            id_list.append(line)
    f.close()
    
    return id_list
    
def smooth(a, hyper=2.5):
    """ Smooth the confidence score
    a中最大为1，最小为0
    """
    import numpy as np
    import math
    num = len(a)
    a = np.array(a, float)
    Z = 0
    for i in range(num):
        a[i] = math.exp(hyper * a[i])
        Z += a[i]
    
    a /= Z
    
    return a
    
def level_distribution(topic_set, topic_level, num_level):
    """ 查看某个topic set中的level的分布
    """
    level_count = [0] * num_level
    for topic_id in topic_set:
        if topic_id in topic_level:
            level = topic_level[topic_id]
            level_count[level] += 1
        else:
            print 'Error: %s in test topic set!' % topic_id
            
    print level_count
    
def load_topic_level(file_path):
    """ 从文件中读入topic id和相对应的真正level
    """
    f = open(file_path, 'r')
    topic_level = dict()
    level_count = [list(), list()]
    y_pred = []
    for line in f:
        line = line.strip()
        seg_list = line.split(' ')
        topic_id = seg_list[0]
        level = int(seg_list[1])
        topic_level[topic_id] = level
        
        level_count[level].append(topic_id)
        y_pred.append(level)
        
    f.close()
    return topic_level, level_count, y_pred
    
def prepare_kong_topics():
    """ 根据group_id为kong的topic list，复制topic信息到data-dynamic/kong/文件夹中
    """
    import os
    path = 'data-dynamic/TopicList-kong-shuffled.txt'
    f = open(path, 'r')
    for topic_id in f:
        topic_id = topic_id.strip()
        path_kong = 'data-dynamic/kong/' + topic_id + '.txt'
        path_qiong = 'data-dynamic/qiong/' + topic_id + '.txt'
        path_buybook = 'data-dynamic/buybook/' + topic_id + '.txt'
        path_zhuangb = 'data-dynamic/zhuangb/' + topic_id + '.txt'
        if os.path.exists(path_qiong):
            args = '/bin/cp ' + path_qiong + ' ' + path_kong
            os.popen(args)
        elif os.path.exists(path_buybook):
            args = '/bin/cp ' + path_buybook + ' ' + path_kong
            os.popen(args)
        elif os.path.exists(path_zhuangb):
            args = '/bin/cp ' + path_zhuangb + ' ' + path_kong
            os.popen(args)
        else:
            print 'Topic: %s not found!' % topic_id
            
    f.close()
