#coding=utf8

"""
确定popularity的范围，作出直方图
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_id_list

# 评论数量的阈值，如果评论数量小于该值，则不考虑
MIN_COMMENT = 50

def get_popularity_level(level, bins, cumprob):
    """
    level: number of levels of popularity
    bins: the bins for each bar
    cumprob: cumulative prob. for the bins
    """    
    prob = [1.0/level] * level
    prob = np.cumsum(prob)
    prob[-1] = 1
    
    bin_index = 0 # index for bins and cumprob
    num_bins = len(bins)
    plevel = [0] * level
    level_index = 0
    
    while True:
        while prob[level_index] > cumprob[bin_index]:
            bin_index += 1
            
        plevel[level_index] = bins[bin_index]
        level_index += 1
        if level_index >= level - 1:
            break
        
        
    plevel[-1] = bins[-1]
    #plevel.insert(0, 0)
    
    return plevel
    
def level_statics(plevel, x):
    """ Print out how many threads each level has
    """
    plevel[-1] = float('inf')
    level_cnt = [0] * len(plevel)
    for t in x:
        index = 0
        while t > plevel[index]:
            index += 1
        level_cnt[index] += 1
        
    print 'Number of threads in each level:', level_cnt

def main(group_id):
    
    topic_list = load_id_list('data-dynamic/TopicList-' + group_id + '.txt')
    print 'Topics id loaded:', len(topic_list)
    
    base_path = 'data-dynamic/' + group_id + '/'
    
    x = [0] * len(topic_list)
    index = 0
    for topic_id in topic_list:
        path = base_path + topic_id + '.txt'
        
        if not os.path.exists(path):
            continue
        
        print 'Processing file: ', path
        f = open(path, 'r')
        #print 'Reading file: ', path
        
        line = f.readline().strip()
        seg_list = line.split('[=]')
        num_comment = int(seg_list[3])
        
        # 过滤掉那些总的评论数小于某个特定值的
        if num_comment < MIN_COMMENT:
            continue
        
        x[index] = num_comment
        index += 1
        
        f.close()
        
    x = x[:index]
    
    xmax = max(x)
    xmin = min(x)
    
    print 'Number of threads:', len(x)
    print 'Max number of comments:', xmax
    print 'Min number of comments:', xmin
    
    (n, bins, patches) = plt.hist(x, bins=range(xmin, xmax, 1), cumulative=True, normed=True)
    #(n, bins, patches) = plt.hist(x, bins=5)
    
    #plt.show()
    
    num_level = 2
    popularity_level = get_popularity_level(num_level, bins, n)
    
    print 'Popularity level: '
    print popularity_level
    
    # tell me how many threads each level has
    level_statics(popularity_level, x)
    
if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    main(group_id)
    #plevel = get_popularity_level(4, range(5), [0.1, 0.3, 0.5, 0.6, 1])
    #print plevel
