import numpy as np

def seq(start,end,step):
    x = []
    temp = start
    while temp <= end:
        x.append(temp)
        temp += step  
    return np.array(x)