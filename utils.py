#encoding=utf8

def seg_chinese(chinese_str):
    """中文分词
    Note: 目前采用jieba分词. jieba项目主页：https://github.com/fxsjy/jieba
    """
    import jieba
    seg_list = jieba.cut(chinese_str)
    return " ".join(seg_list)
    
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
    
def transform_ts(p, alpha = 1.2, T_smooth = 3):
    """ 按照NIPS'13文章中的预处理方法处理单个ts
    alpha: how to emphasize the large spikes
    T_smooth: the smoothing parameter
    """
    import numpy as np
    import math
    count = len(p)
    # extend the array
    p = np.array(p, float)
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
    
