#coding=utf8

import numpy as np
import matplotlib.pyplot as plt

def pubdate_distribution(group_id, dataset):
    """ 考察静态因素：帖子发表时间和level的分布
    Note: 考察在全天24小时内，不同level的帖子的分布
    """
    p = [list(), list()]
    for topic_id, topic_feature, popularity_level in dataset:
        pubdate = topic_feature[0][5]
        hour_index = pubdate.hour
        p[popularity_level].append(hour_index)
        
    plt.figure()
    
    n, bins, patches = plt.hist([p[0], p[1]], 24, normed=1, histtype='bar')
    # Calculates a Pearson correlation coefficient and the p-value for testing non-correlation.
    import scipy.stats
    corr_coef, pvalue = scipy.stats.pearsonr(n[0], n[1])
    print 'Correlation coef: %f, P-value: %f' % (corr_coef, pvalue)    
    plt.title('Factor: pubdate. Correlation coef: %f, P-value: %f' % ( corr_coef, pvalue))
    #import ipdb; ipdb.set_trace()
    plt.show()
    
        
def factor_propagation_plot(group_id, dataset, num_feature, category_count_list, factor_index_list, factor_name):
    """ 此图用于查看那些具有类似早期propagation模式的帖子是否具有类似的popularity
    factor_index_list: 每个factor在feature中的index list
    factor_name：相对应的factor的name list
    横坐标：时间
    纵坐标：dynamic factor的变化
    Note: 不同的深浅的颜色标注不同的popularity level
    """
    num_factor = len(factor_index_list)
    num_level = len(category_count_list) # 这里只是用到两个level
    colors = ['r', 'g', 'b', 'y', 'k']
    for i in range(num_factor):
        #import ipdb; ipdb.set_trace()
        factor_index = factor_index_list[i]
        p0 = np.zeros((category_count_list[0], num_feature), float)
        p1 = np.zeros((category_count_list[1], num_feature), float)
        level_index = [0, 0]
        for topic_id, topic_feature, popularity_level in dataset:
            tmp = [topic_feature[j+1][factor_index] for j in range(num_feature)]
            tmp = np.array(tmp, float)   
            index = level_index[popularity_level]
            if popularity_level == 0:
                p0[index, :] = tmp
            else:
                p1[index, :] = tmp
            level_index[popularity_level] += 1
        
        # get the mean and std    
        mean0 = np.mean(p0, axis=0)
        std0 = np.std(p0, axis=0)
        
        mean1 = np.mean(p1, axis=0)
        std1 = np.std(p1, axis=0)
        
        plt.title('Group:%s, Dynamic factor:%s' % (group_id, factor_name[i]))
        plt.errorbar(range(num_feature), mean0, yerr=std0, c='r', hold=True)
        plt.errorbar(range(num_feature), mean1, yerr=std1, c='b', hold=True)
        
        plt.show()
        
def factor_propagation_plot_old(dataset, num_feature):
    """ 此图用于查看那些具有类似早期propagation模式的帖子是否具有类似的popularity
    横坐标：时间
    纵坐标：dynamic factor的变化
    Note: 不同的深浅的颜色标注不同的popularity level
    """
    num_factor = 1
    num_level = 3
    colors = ['r', 'g', 'b', 'y', 'k']
    
    for i in range(num_factor):
        #import ipdb; ipdb.set_trace()
        level_cnt_list = [0] * num_level
        p = np.zeros((num_level, num_feature), float)
        for topic_id, topic_feature, popularity_level in dataset:
            tmp = [topic_feature[j+1][i] for j in range(num_feature)]
            tmp = np.array(tmp, float)   
            p[popularity_level, :] = p[popularity_level, :] + tmp
            level_cnt_list[popularity_level] += 1
        
        ax = plt.subplot(111)
        for j in range(num_level):
            p[j, :] = p[j, :] * 1.0 / level_cnt_list[j]
            ax.plot(p[j, :], c = colors[j])
            
        plt.show()
        
def topic_propagation_plot(dataset, num_feature):
    """ 画出每个topic的每个factor随着时间的变化，依据分类情况
    """
    num_factor = 3
    num_level = 3
    colors = ['r', 'g', 'b', 'y', 'k']
    
    for topic_id, topic_feature, popularity_level in dataset:
        ax = plt.subplot(111)
        ax.set_title('Topic id: %s, Type: %d' % (topic_id,popularity_level))
        for i in [0,1,2]:
            num_feature = len(topic_feature) - 1
            tmp = [topic_feature[j+1][i] for j in range(num_feature)]
            tmp = np.array(tmp, float)
            #tmp = tmp * 1.0 / max(tmp)
            ax.plot(tmp, c=colors[i])
        plt.show()
