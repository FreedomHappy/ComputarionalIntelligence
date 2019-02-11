# encoding: utf-8
"""
@author: lin
@file: PSO.py
@time: 2018/11/15 14:05
@desc:
"""
"""粒子群优化算法PSO
1.初始化所有粒子
初始化粒子的速度和位置向量
历史最优pBest设为当前位置，群体最优gBest设为当前最优粒子
2.适应度函数值
3.历史最优判断
4.全局最优判断
5.更新速度和位置向量
位置更新需合法
6.结束条件
"""
import numpy as np
import random
import matplotlib.pyplot as plt
"""粒子类"""
class Particle:
    def __init__(self,dimension):
        self.dimension = dimension
        self.V = np.random.rand(dimension)*20 + (-10)
        self.X = np.random.rand(dimension)*20 + (-10)
    def setV(self,V):
        self.V = V
    def getV(self):
        return self.V
    def setX(self, X):
        self.X = X
    def getX(self):
        return self.X
    def setpBest(self,best):
        self.pBest =  best
    def getpBest(self):
        return self.pBest
"""粒子群"""
class ParticleSwarm:
    def __init__(self):
        self.solutionSpace = np.array([-10.0,10.0]) # 解空间
        self.Vmax = self.solutionSpace*0.2 # 速度限制 一般取相应维的10%-20%
        self.N = 3 # 粒子数
        self.dimension = 2 # X,V 维度
        self.iter_num = 100 # 迭代次数
        self.weight = 0.9 #惯量权重
        self.c1 = 2.0 # 加速系数，学习因子
        self.c2 = 2.0 # 加速系数，学习因子
        # 初始化群体各粒子
        self.ps = []
        result = []
        for i in range(self.N):
            # 初始化粒子的X,V
            p = Particle(self.dimension)
            # 初始化pBest
            r = self.fitness(p.getX())
            p.setpBest(p.getX())
            self.ps.append(p)
            result.append(r)
        self.gBest = self.ps[int(np.argmin(result))].getX()

    def fitness(self, X):
        X = X**2
        result = np.sum(X)
        return result
    def runPSO(self):
        iter_num = self.iter_num
        ps = self.ps
        self.update(ps)
        for i in range(iter_num):
            print("第{}次迭代".format(i+1))
            self.update(ps)
            print(self.gBest)
            print(self.fitness(self.gBest))
            for p in ps:
                print(p.getX())
    def update(self,ps):
        gBest = self.gBest
        result = []
        for p in ps:
            result.append(self.updateParticle(p))
        if np.min(result) < self.fitness(gBest):
            self.gBest = ps[np.argmin(result)].getX()
    def updateParticle(self,p):
        gBest = self.gBest
        pBest = p.getpBest()
        oldV = p.getV()
        oldX = p.getX()
        rand1 = random.random()
        rand2 = random.random()
        # 公式1
        newV =self.weight*oldV +  \
           self.c1*rand1*(pBest-oldX) + \
           self.c2*rand2*(gBest-oldX)
        # 修正V
        newV = self.correctV(newV)
        # 公式2
        newX = p.getX() + newV
        # 修正X
        newX = self.correctX(newX)
        # 更新V，X
        p.setV(newV)
        p.setX(newX)
        # 更新pBest
        result = self.fitness(newX)
        if result < self.fitness(pBest):
            p.setpBest(newX)
        return result
    def correctX(self,X):
        sSpace = self.solutionSpace
        X[X<sSpace[0]]=sSpace[0]
        X[X>sSpace[1]]=sSpace[1]
        return X
    def correctV(self,V):
        Vmax = self.Vmax
        V[V<Vmax[0]]=Vmax[0]
        V[V>Vmax[1]]=Vmax[1]
        return V
def testPSO():
    PS = ParticleSwarm()
    PS.runPSO()

if __name__=="__main__":
    testPSO()