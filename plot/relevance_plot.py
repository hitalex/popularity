#coding=utf8

"""
功能：
popularity和dynamic factor的相关关系图
"""
import os.path
from datetime import datetime,timedelta

import numpy as np
import matplotlib.pyplot as plt

from dynamic_feature import dynamic_factor_dict
from utils import load_id_list

# 在帖子发表 AFTER_PUBLISHING_TIME 时间后开始记录dynamic factor的值
AFTER_PUBLISHING_TIME = timedelta(days=0.5)

def factor_relevance_plot(popularity, factor):
    """ 根据对应的factor值和popularity值，作出log-log图，查看其相关关系
    """
    #plt.loglog(factor, popularity, basex=10, basey=10, ls='-.', color='b')
    popularity = np.array(popularity, int)
    factor = np.array(factor, float)
    
    # 确保popularity和factor中的所有元素都大于等于0
    popularity = np.log(popularity, dtype='float64')
    factor = np.log(factor, dtype='float64')
    
    fig = plt.figure()
    plt.scatter(factor, popularity, c='blue')
    
    plt.show()
    

def main(group_id, factor_index):
    topic_list_path = 'data-dynamic/TopicList-' + group_id + '.txt'
    topic_list = load_id_list(topic_list_path)
    print 'Num of topics loaded:', len(topic_list)
    
    popularity = [0] * len(topic_list)
    factor = [0] * len(topic_list)
    index = 0
    base_path = 'data-dynamic/' + group_id + '/'
    for topic_id in topic_list:
        path = base_path + topic_id + '.txt'
        if not os.path.exists(path):
            continue
        
        f = open(path, 'r')
        try:
            # get the thread publish date
            line = f.readline().strip()
            if line == '':
                f.close()
                continue
                
            seg_list = line.split('[=]')
            thread_pubdate = datetime.strptime(seg_list[2], '%Y-%m-%d %H:%M:%S')
            
            curr_comment_cnt = 0
            for line in f:
                line = line.strip()
                seg_list = line.split('[=]')
                pubdate = datetime.strptime(seg_list[2], '%Y-%m-%d %H:%M:%S')
                curr_comment_cnt += 1
                
                if pubdate < thread_pubdate + AFTER_PUBLISHING_TIME:
                    continue
                    
                factor_value = float(seg_list[factor_index])
                
                popularity[index] = curr_comment_cnt
                factor[index] = factor_value
                index += 1
                break
        except Exception as e:
            print 'Exception occured:', e
            print 'Errors in topic:', topic_id
        finally:
            f.close()
        
    
    print 'Number of pairs:', index
    popularity = popularity[:index]
    factor = factor[:index]
    
    factor_relevance_plot(popularity, factor)

if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    print 'Choose dynamic factor:'
    print dynamic_factor_dict
    # dynamic factor index
    factor_index = int(sys.argv[2])
    main(group_id, factor_index)
