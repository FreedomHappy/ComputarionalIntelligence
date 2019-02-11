# encoding: utf-8
"""
@author: lin
@file: som.py
@time: 2018/9/22 21:54
@desc:
"""

import numpy as np
import random
import matplotlib.pyplot as plt


class SOM(object):
    def __init__(self, n, m, lr):
        self.n = n
        self.m = m
        self.old_lr = lr
        self.lr=lr
        print('lr',self.lr)
        pass

    def ComptitiveLearning(self, inputs):
        self.Neig = 2

        # 归一化输入，初始化权重
        new_inputs = []
        for input in inputs:
            new_inputs.append(self.normalize(input))
        x = new_inputs
        self.init_weights(x)
        # self.weights=np.random.rand(self.n, self.m, inputs.shape[1])

        #开始更新
        epoch = 0
        while self.lr>0.000001 and epoch<1000:
            epoch = epoch+1
            random.shuffle(new_inputs)
            # print('new_inputs:', new_inputs)
            self.lr_decay(epoch)
            self.update_Neig(epoch)
            for new_input in new_inputs:
                self.update(new_input)
            print('epoch %d complete'%(epoch))
            if epoch % 200 ==0:
                print('weights:\n', self.weights)
        print('结束！')
        res=self.getDict_two(inputs)
        return res
    def getDict_two(self,inputs):
        res = {}
        N, M, _ = np.shape(self.weights)
        for i in range(len(inputs)):
            n, m = self.winner_euc(inputs[i])
            key = n * M + m
            if key in res:
                res[key].append(i)
            else:
                res[key] = []
                res[key].append((n, m))
                res[key].append(i)
        return res



    def getDict_one(self, inputs):
        res = {}
        N, M, _ = np.shape(self.weights)
        for i in range(len(inputs)):
            n, m = self.winner_euc(inputs[i])
            key = n * M + m
            if key in res:
                res[key].append(i)
            else:
                res[key] = []
                res[key].append(i)
        return res


    def update(self, input):
        maxrow, maxcol = self.winner_euc(input)
       # print('row,col:', maxrow, maxcol)
        self.update_wimmers(maxrow, maxcol, input)
        pass


    def update_Neig(self, epoch):
         if self.Neig==0: return
         # size = self.epochs_size // self.Neig
         self.Neig = self.Neig - epoch // 500

    def lr_decay(self, epoch):
        self.lr = -(0.46/1000)*epoch+self.old_lr
        self.lr = self.lr*np.exp(-10)
        # self.lr=self.old_lr / (1 + epoch / 1000)
        print('lr',self.lr)

    def update_wimmers(self, cen_r, cen_l, input):
        for i in range(self.n):
            for j in range(self.m):
                if (abs(i-cen_r) <= self.Neig) or (abs(j-cen_l) <= self.Neig):
                    self.up_winnner_par(input, i, j)
                    pass

    def up_winnner_par(self, input, i, j):
        # print('update wiehgts')
        # print('old w', self.weights[i][j])
        #print(self.lr)
        self.weights[i][j] = self.weights[i][j]+self.lr*(input-self.weights[i][j])
        # print('new w', self.weights[i][j])

    def init_weights(self, x):
        random.shuffle(x)
        self.weights = []
        for i in range(self.n):
            weight=[]
            for j in range(self.m):
                weight.append(x[j % len(x)])
            self.weights.append(weight)



    def normalize(self, x):
        y = np.sqrt(np.dot(x, x.T))
        return x/y

    def winner_euc(self, x):
        result=[]
        for weights in self.weights:
            result.append(np.dot(weights, x.T))
        result=np.array(result)
        result.reshape(self.n, self.m)
        # print('result:', result)
        li=np.argmax(result, axis = 1)
        # print('li',li)
        r = []
        for i, j in zip(range(self.n), li):
            r.append(result[i][j])
        # print('r:', r)
        i = np.argmax(r)
        return i, li[i]


def draw_one(C , dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm' , 'd']
    count = 0
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[count % len(color)], label=i)
        count += 1
    plt.legend(loc='upper right')
    plt.show()

def draw_two(C, dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm', 'd']
    count = 0
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        X.append(datas[0][0])
        Y.append(datas[0][1])
        # for j in range(len(datas))[1:]:
        #     print(i, datas[j])
        #     X.append(datas[j])
        #     Y.append(datas[j])
        lab=[]
        for j in datas[1:]:
            lab.append(dataSet[j])
        plt.scatter(X, Y, marker='o', color=color[count % len(color)], label=lab)
        count += 1
    plt.legend(loc='best')
    plt.show()


def main():
    inputs=np.array([[1, 0, 0, 0],[1, 1, 0, 0],[1, 1, 1, 0],[0, 1, 0, 0],[1, 1, 1, 1]])
    som=SOM(5, 5, 0.5)
    res=som.ComptitiveLearning(inputs)
    print(res)
    #print(res[3][0])
    draw_two(res, inputs)

if __name__=="__main__":
    main()