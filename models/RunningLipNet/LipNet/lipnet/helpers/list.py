def get_list_safe(l, index, size): # list, index, size를 받음.
    ret = l[index:index+size] # ret = list를 index~ index+size 범위로 slice
    while size - len(ret) > 0: 
        ret += l[0:size - len(ret)]
    return ret