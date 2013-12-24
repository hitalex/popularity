#coding=utf8

"""
整理数据格式：目标为每个项目整理成一行
"""
import os.path

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
    
def remove_newline(spath, tpath):
    fs = open(spath, 'r')
    ft = open(tpath, 'w')
    
    for line in fs:
        line = line.replace('\r','')
        line = line.replace('\n','')
        line = line.strip()
        if line != '[*ROWEND*]':
            ft.write(line + ' ')
        else:
            ft.write('\n')
    
    fs.close()
    ft.close()
    
if __name__ == '__main__':
    # remove topic content desc
    gid = 'qiong'
    
    topic_list = load_id_list('/home/kqc/dataset/douban-group/' + gid + '/' + gid + '-TopicList.txt')
    
    source_path = '/home/kqc/dataset/douban-group/' + gid + '/'
    target_path = '/home/kqc/dataset/new-douban-group/' + gid + '/'
    
    spath = source_path + gid + '-info.txt'
    tpath = target_path + gid + '-info.txt'
        
    remove_newline(spath, tpath)
    
    for tid in topic_list:
        spath = source_path + tid + '-content.txt'
        tpath = target_path + tid + '-content.txt'
        
        if not os.path.exists(spath):
            print 'File %s does not exist!' % spath
            continue
        
        remove_newline(spath, tpath)
        
        spath = source_path + tid + '-comment.txt'
        tpath = target_path + tid + '-comment.txt'
        
        remove_newline(spath, tpath)
