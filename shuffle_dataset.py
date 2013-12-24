#coding=utf8

"""
对topiclist中的topic进行shuffle
"""
import random

from utils import load_id_list

def main():
    group_id = 'qiong'
    topiclist_path = 'data-dynamic/TopicList-' + group_id + '-shuffled.txt'
    topic_list = load_id_list(topiclist_path)
    random.shuffle(topic_list)
    
    topiclist_path = 'data-dynamic/TopicList-' + group_id + '-shuffled.txt'
    f = open(topiclist_path, 'w')
    for topic_id in topic_list:
        f.write(topic_id + '\n')
    f.close()

if __name__ == '__main__':
    main()
