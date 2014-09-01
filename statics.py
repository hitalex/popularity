#coding=utf8

"""
对数据集进行数据统计，包括评论数、帖子生命周期、如何选取gaptime和delta t（t_t - t_r）
"""
import os

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
font = FontProperties(fname=r"/usr/share/fonts/WinFonts/simsun.ttc", size=10) 

from datetime import datetime, timedelta
from utils import load_id_list

import numpy as np

# 评论数量的阈值，如果评论数量小于该值，则不考虑
MIN_COMMENT = 10
# 评论数量的最大值
MAX_COMMENT = 1000
# 可以看作是viral的最少评论数, 目前不打算在此进行约束
# 目标时刻的评论数用threshold来约束
VIRAL_MIN_COMMENT = 50
# 抓取内容的时间。如果target_date在此之后，则同样不进行预测
#DEADLINE = datetime(2013, 11, 15)
DEADLINE = datetime(2014, 5, 20) # for tianya forum
# 在开始预测时，最少需要拥有的comment数量
MIN_COMMENT_PREDICTION_DATE = 10

MAX_LIFECYCLE = 10000

def plot_histogram(x, title):
    """ 根据lifespan的list画出帖子生命周期的分布直方图
    """
    xmin = int(min(x))
    xmax = int(max(x))
    
    n, bins, patches = plt.hist(x, bins=range(xmin, xmax+1, 1), normed=0, facecolor='green', alpha=0.5)
    plt.show()
    
def plot_loglog(ax, data, title = u'', xlabel_text = u'', ylabel_text = u''):
    """ 根据数据分布，画出log-log图
    """
    from scipy.stats import itemfreq
    
    tmp = itemfreq(data) 
    x = tmp[:, 0] # x 已经被排好序了
    y = tmp[:, 1]
    
    """
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(x, y, marker='.', color='b')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)
    #ax.title(title)
    """
    ax.loglog(x, y, basex=2, basey=2, ls='None', marker='x', color='b')
    ax.grid(True)
    ax.set_title(title, fontproperties=font)
    ax.set_xlabel(xlabel_text, fontproperties=font)
    ax.set_ylabel(ylabel_text, fontproperties=font)
    

def collect_comments_lifecycle(group_id, topic_list):
    """ 收集评论数和持续时间长度
    NOTE：这里只计算帖子已经持续存在的时间，无法计算生命周期长度
    """
    num_comment_list = []   # 评论数列表
    num_lifecycle_list= []  # 生命周期长度列表
    seconds_in_a_day = 60 * 60 * 24;
    
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
        
        if total_comment < MIN_COMMENT:
            continue
        
        last_comment_date = None # 最后一个评论的发表时间
        for line in f:
            line = line.strip()
            # eval the str as feature dict
            nan = float('nan')
            feature_dict = eval(line)
            # basic info
            cid                     = feature_dict['cid']
            pid                     = feature_dict['pid']
            pubdate                 = datetime.strptime(feature_dict['pubdate'], '%Y-%m-%d %H:%M:%S')
            
            last_comment_date = pubdate
        
        f.close()
        
        delta_time = last_comment_date - thread_pubdate
        num_days = delta_time.total_seconds() * 1.0 / seconds_in_a_day # 计算天数
        
        # 过滤出持续时间大于一定时间的帖子
        if num_days > MAX_LIFECYCLE:
            print 'Filtered: %s, num_days: %f' % (topic_id, num_days)
            continue
        
        num_comment_list.append(total_comment)
        num_lifecycle_list.append(num_days)
        
    return num_comment_list, num_lifecycle_list

def main(group_id):
    import pickle
    #topiclist_path = 'data-dynamic/TopicList-' + group_id + '-shuffled.txt'
    topiclist_path = 'data-dynamic/' + group_id + '-post-list.txt' # for Tianya dataset
    topic_list = load_id_list(topiclist_path)
    print 'Number of total topics loaded: ', len(topic_list)
    
    """
    # 存储中间结果
    num_comment_list, num_lifecycle_list = collect_comments_lifecycle(group_id, topic_list)    
    print 'Number of threads:', len(num_comment_list)
    f = open('pickle/comment-lifecycle-dist-tianya.pickle', 'w')
    pickle.dump([num_comment_list, num_lifecycle_list], f)
    f.close()
    """
    
    f = open('pickle/comment-lifecycle-dist-tianya.pickle', 'r')
    num_comment_list, num_lifecycle_list = pickle.load(f)
    f.close()
    
    #import ipdb; ipdb.set_trace()
    #plot_histogram(num_comment_list, '')    
    #plot_histogram(num_lifecycle_list, '')
    
    fig = plt.figure()
    ax1 = plt.subplot(121) # 左边的图
    ax2 = plt.subplot(122) # 右边的图
    
    print 'Number of elements: ', len(num_comment_list)
    #plot_loglog(num_comment_list, u'', u'评论数', u'讨论帖数量')
    plot_loglog(ax1, num_comment_list, '', 'Number of comments', 'Number of threads')
    
    
    for i in range(len(num_lifecycle_list)):
        #num_lifecycle_list[i] = int(num_lifecycle_list[i] * 24)
        num_lifecycle_list[i] = int(num_lifecycle_list[i])
     
    #plot_loglog(num_lifecycle_list, u'', u'生命周期长度', u'讨论帖数量')
    plot_loglog(ax2, num_lifecycle_list, '', 'Length of lifecycle(days)', 'Number of threads')
    
    plt.show()

if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    main(group_id)
