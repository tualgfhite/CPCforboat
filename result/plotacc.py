# -*- coding:utf-8 -*-

import re

import numpy as np
from matplotlib import pyplot as plt



def smooth(data, ss=1):
    if ss > 1:
        y = np.ones(ss) / float(ss)
        data = np.hstack((data, data[1 - ss:]))
        data = np.convolve(data, y, "valid")
    return data

def plot(path,savepath,ss):
    with open(path) as f:
        temp = f.read()
    result = []
    for i in range(39, len(temp)):
        result.append(temp[i][-22:-16])
    pattern='(?<=Accuracy:\s)\d\.\d{4}'
    res=re.findall(pattern,temp)
    acc=[float(x)for x in res]
    plt.plot(smooth(np.array(acc), ss))
    plt.ylabel("acc")
    plt.xlabel("steps")
    plt.title(f"{savepath} acc")
    plt.savefig(f"{savepath}.png", bbox_inches='tight')
    plt.close()
if __name__ =="__main__":
    apath='cpc/cpc-2022-07-13_21_29_40.log'
    bpath='cpc/cpc-2022-07-14_16_14_32.log'
    plot(apath,'pretrain',10)
    plot(bpath,'classfier',20)
