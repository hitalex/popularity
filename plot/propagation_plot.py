#coding=utf8

"""
对于每个帖子，考察其dynamic factor随着comment数量增加的变化，目的是找出那些具有明显的
正相关和负相关的dynamic factor
"""

import os.path
from datetime import datetime,timedelta

import numpy as np
import matplotlib.pyplot as plt

from dynamic_feature import dynamic_factor_dict, dynamic_factor_list
from utils import load_id_list

def propagation_plot(factor_value, factor_index_list):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    ax = plt.subplot(111)
    
    num_factor, num_comment = factor_value.shape
    if num_factor > len(colors):
        print 'Warning: not engough colors'
        
    for i in range(num_factor):
        color_index = i % len(colors)
        factor_index = factor_index_list[i]
        factor_name = dynamic_factor_list[factor_index - 4]
        
        ax.plot(factor_value[i, :], c=colors[color_index], label=factor_name)
        
    ax.legend()
    plt.show()

def main(group_id):
    topic_list_path = 'data-dynamic/TopicList-filtered-' + group_id + '.txt'
    topic_list = load_id_list(topic_list_path)
    print 'Num of topics loaded:', len(topic_list)
    
    base_path = 'data-dynamic/' + group_id + '/'
    factor_index_list = [4, 9, 10, 11] # 需要考察的factor变量
    for topic_id in topic_list:
        path = base_path + topic_id + '.txt'
        if not os.path.exists(path):
            continue
        
        f = open(path, 'r')
        try:
            line = f.readline().strip()
            seg_list = line.split('[=]')
            comment_cnt = int(seg_list[3]) # 总的comment数目
            
            # 记录各个dynamic factor的变化
            factor_value = np.zeros((len(factor_index_list), comment_cnt), float)
            index = 0 # comment的index
            for line in f:
                line = line.strip()
                seg_list = line.split('[=]')
                
                for i in range(len(factor_index_list)):
                    factor_index = factor_index_list[i]
                    factor_value[i, index] = float(seg_list[factor_index])
                
                index += 1
            
        except Exception as e:
            print 'Exception occured:', e
            print 'Errors in topic:', topic_id
        finally:
            f.close()
            
        propagation_plot(factor_value, factor_index_list)

if __name__ == '__main__':
    import sys
    group_id = sys.argv[1]
    
    print 'Choose dynamic factor:'
    print dynamic_factor_list
    # dynamic factor index
    #factor_index = int(sys.argv[2])
    main(group_id)
