#coding=utf8

"""
DTW algorithm for test
"""

import numpy as np
import numpy.linalg

def get_cost(p1, p2):
    """ Get the cost between two points
    """
    return abs(p1 - p2)

def DTW_distance(s, t):
    """ DTW distance for s and t
    Ref: http://en.wikipedia.org/wiki/Dynamic_time_warping
    """
    s.insert(0, 0)
    t.insert(0, 0)
    
    sl = len(s)
    tl = len(t)
    # DTW distance array
    dis = np.zeros((sl, tl), np.float64)
    
    dis[:, 0] = float('inf')
    dis[0, :] = float('inf')
    
    dis[0, 0] = 0
    
    for i in range(1, sl):
        for j in range(1, tl):
            cost = get_cost(s[i], t[j])
            dis[i, j] = cost + min(dis[i-1, j], dis[i, j-1], dis[i-1, j-1])
            
    return dis[sl-1, tl-1]
    
def Euclidean_distance(r, s):
    """ The Euclidean distance
    Note: r and s must be of the same length
    """
    assert(len(r) == len(s))
    r = np.array(r, float)
    s = np.array(s, float)
    dis = np.linalg.norm(r - s)
    
    return dis
    
def best_match_distance(r, s, delta_max=-1):
    """ The distance metric proposed in NIPS'13 and SDM'11
    r: the longer time series
    s: the shorter time series
    delta_max: width of sliding window
    """
    if len(r) < len(s):
        tmp = r
        r = s
        s = tmp
        
    delta = len(r) - len(s)
    if delta_max < 0:
        delta_max = delta

    if delta_max > delta:
        delta_max = delta

    min_dis = float('inf')
    T = len(s)
    best_shift = -1
    for i in range(0, delta_max):
        tmp = r[i:i+T]
        dis = Euclidean_distance(tmp, s)
        if dis < min_dis:
            min_dis = dis
            best_shift = i
            #print min_dis, best_shift
        
    #print 'Best shift:', best_shift
    return min_dis

def main():
    s = [1, 2, 4, 5, 10, 3]
    t = [4, 5, 3]
    
    #print DTW_distance(s, t)
    print best_match_distance(s, t, 4)

if __name__ == '__main__':
    main()
