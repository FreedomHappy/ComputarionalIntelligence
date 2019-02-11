# encoding: utf-8
"""
@author: lin
@file: CompetitiveLearning.py
@time: 2018/9/22 16:29
@desc:
"""
import numpy as np


class SOM(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [np.random.randn()for x, y in zip(sizes[1:], sizes[:-1])]
        pass

    def __normalize(self, x):
        sum=0
        for i in x:
            sum+= i*i
        y = np.sqrt(sum)
        z = []
        for i in x:
            z.append(i/y)
        return z

    def __winner_euc(self, x):
        result=[]
        for layer_weights in self.weights:
            layer_result = [np.dot(x, weight) for weight in layer_weights]
            j = np.argmax(layer_result, 0)
            result.append([j, layer_result[j]])
        i = np.argmax([layer[1] for layer in result],0)
        return i, result[i][0]
