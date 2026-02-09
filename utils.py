import numpy as np
pi = 3.1415

def seq(start,end,step):
    x = []
    temp = start
    while temp <= end:
        x.append(temp)
        temp += step  
    return np.array(x)

def pnorm(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)