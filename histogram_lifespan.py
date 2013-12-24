#coding=utf8

"""
统计每个帖子的生命周期，并作出直方图
标准：如果一个帖子在30天之内没有新的评论，则可以认为此帖已经死掉
"""
import os.path
import os

import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from utils import load_id_list

# 设置11月初为截至时间
DEADLINE = datetime(2013, 11, 1)
# 如果隔MAX_SILENT_TIME 这么长时间没有人回复，则认为帖子已经死掉
MAX_SILENT_TIME = timedelta(days=30)

seconds_in_one_day = 24 * 60 * 60

def plot_histogram(lifespan_list):
    """ 根据lifespan的list画出帖子生命周期的分布直方图
    """
    
    x = [item.total_seconds() * 1.0/ seconds_in_one_day for item in lifespan_list]
    xmin = int(min(x))
    xmax = int(max(x))
    
    n, bins, patches = plt.hist(x, bins=range(xmin, xmax, 1), normed=0, facecolor='green', alpha=0.5)
    plt.show()

def main(group_id):
    latest_comment_time = DEADLINE - MAX_SILENT_TIME
    
    topic_list_path = '/home/kqc/dataset/douban-group/TopicList-' + group_id + '.txt'
    topic_list = load_id_list(topic_list_path)
    print 'Num of topics loaded:', len(topic_list)
    
    lifespan_list = [0] * len(topic_list)
    index = 0
    base_path = '/home/kqc/dataset/douban-group/' + group_id + '/'
    for topic_id in topic_list:
        path = base_path + topic_id + '-info.txt'
        if not os.path.exists(path):
            continue
        
        # get the last line of a file
        line = os.popen("tail -1 " + path).readlines()[0]
        line = line.strip()
        if line == '':
            continue
        seg_list = line.split('[=]')
        # 最后一个comment的发布时间
        last_comment_pubdate = datetime.strptime(seg_list[4], '%Y-%m-%d %H:%M:%S')
        if last_comment_pubdate > latest_comment_time:
            continue
        
        # get the first line
        line = os.popen("head -1 " + path).readlines()[0]
        line = line.strip()
        seg_list = line.split('[=]')
        thread_pubdate = datetime.strptime(seg_list[4], '%Y-%m-%d %H:%M:%S')
        
        #if total_comment < MIN_COMMENT or DEADLINE < thread_pubdate + target_date:
        #    continue
        
        lifespan = last_comment_pubdate - thread_pubdate
        
        # 如果生命周期大约30天，则不考虑
        if lifespan.total_seconds() > 90 * seconds_in_one_day:
            continue
        
        lifespan_list[index] = lifespan
        index += 1
        
        
    lifespan_list = lifespan_list[:index]
    
    plot_histogram(lifespan_list)

if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    main(group_id)
