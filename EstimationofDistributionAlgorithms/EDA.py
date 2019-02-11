# encoding: utf-8
"""
@author: lin
@file: EDA.py
@time: 2018/11/18 11:17
@desc:paper《改进的正态分布的分布估计算法》-《计算机科学》(ComputerScience),2015年8月
摘　要　针对连续空间函数优化问题，提出了改进的正态分布的分布估计算法。该算法将优选出的个体看作正态分
布，然后以正态分布概率模型随机采样产生新的种群，并挑选部分个体与保留的最好解进行交叉操作。将其与均匀分
布的分布估计算法、正态分布的分布估计算法进行了比较，结果证明该方法的效果更好。最后分析了选择较好个体的
比例对算法的影响。
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
"""
一、初始化
    （1）生成初始种群
    （2）评估种群适应度
二、循环
    （1）选择操作
    （2）构建概率模型
    （3）抽样产生新种群
    （4）评估种群适应度
"""
"""测试函数
F1
"""
class EDANormal:
    def __init__(self):
        self.indiv_num = 1000 # 种群总个数
        self.sel_num = int(0.6 * self.indiv_num)  # 挑选的个数
        self.Pc = 0.01
        self.precision = 0.000001
        self.iter_num = 100
        self.F = [-1, 1, 30]  # F1函数
        self.dimension = self.F[2]  # n维
        self.population = None # 种群
    def runEDA(self,FunNum,is_improve):
        if FunNum==1:
            if is_improve==False:
                answer, itern=self.oldEDA_Normal(self.fitness1)
            else:
                answer, itern=self.modifyEDA_Normal(self.fitness1)
        elif FunNum==2:
            if is_improve==False:
                answer, itern=self.oldEDA_Normal(self.fitness2)
            else:
                answer, itern=self.modifyEDA_Normal(self.fitness2)
        elif FunNum==3:
            if is_improve==False:
                answer, itern=self.oldEDA_Normal(self.fitness3)
            else:
                answer, itern=self.modifyEDA_Normal(self.fitness3)
        elif FunNum==4:
            if is_improve==False:
                answer, itern=self.oldEDA_Normal(self.fitness4)
            else:
                answer, itern=self.modifyEDA_Normal(self.fitness4)
        else:
            print("无此函数项")
            answer, itern=None,None
        return  answer,itern
    # 初始化种群
    def __initPopu(self):
        population = np.zeros((self.dimension, self.indiv_num))
        for i in range(self.dimension):
            # 对于Xi要均匀分布
            p = np.random.uniform(low=self.F[0], high=self.F[1], size=[self.indiv_num])
            population[i] += p
        self.population = population.T
    def oldEDA_Normal(self,fitFun):
        bestIndiv = np.ones(self.dimension)*np.inf # 运行过程中的最优解
        answer = [] # 每次迭代的最优解
        itern = 0 # 计算迭代次数
        self.__initPopu()
        # for iter in range(iter_num):
        while 1:
            # print("第{}次迭代".format(iter))
            # 适应度函数评估并排序
            itern +=1
            fitLis,values = self.sortPopu(self.population,fitFun)
            bestIdx = int(fitLis[0])
            thisans = fitFun(self.population[bestIdx])
            answer.append(thisans)
            if fitFun(bestIndiv) > thisans:
                bestIndiv = self.population[ bestIdx]
            if thisans < self.precision:
                break
            # print(np.mean(values))
            # 获取均值和方差
            mu,sigma=self.get_muANDsigma(self.population,fitLis[:self.sel_num])
            # 产生新种群
            self.population = self.generatePopu(mu,sigma,self.indiv_num)
        return answer,itern
    def modifyEDA_Normal(self,fitFun):
        bestIndiv = np.ones(self.dimension) * np.inf  # 运行过程中的最优解
        answer = []  # 每次迭代的最优解
        itern = 0  # 计算迭代次数
        self.__initPopu()
        # for iter in range(iter_num):
        while 1:
            # print("第{}次迭代".format(iter))
            # 适应度函数评估并排序
            itern += 1
            fitLis, values = self.sortPopu(self.population, fitFun)
            bestIdx = int(fitLis[0])
            thisans = fitFun(self.population[bestIdx])
            answer.append(thisans)
            if fitFun(bestIndiv) > thisans:
                bestIndiv = self.population[bestIdx]
            if thisans < self.precision:
                break
            # print(np.mean(values))
            # 获取均值和方差
            mu, sigma = self.get_muANDsigma(self.population, fitLis[:self.sel_num])
            # 产生新种群
            self.population = self.generatePopu(mu, sigma, self.indiv_num)
            # 交叉算子
            self.population=self.cross(self.Pc,self.population,bestIndiv)
        return  answer,itern

    # 获取适应度排序
    def sortPopu(self,population,fitFun):
        fitValue = map(fitFun,population)
        fitDict = {}
        for idx ,value in enumerate(list(fitValue)):
            fitDict[idx]=value
        lis = sorted(fitDict.items(),key=lambda item:item[1])
        results = []
        values = []
        for l in lis:
            results.append(l[0])
            values.append(l[1])
        return  results,values
    # 函数1
    def fitness1(self,p):
        return np.sum(p**2)
    # 函数2
    def fitness2(self,p):
        mat = np.matrix(abs(p))
        return np.sum(abs(p))+np.sum(mat*mat.T)
    # 函数3
    def fitness3(self,p):
        return np.max(abs(p))
    # 函数4
    def fitness4(self,p):
        sum = 0
        for i in range(len(p)):
            sum += (np.sum(p[0:i+1]))**2
        return sum
    # 获取mu和sigma
    def get_muANDsigma(self,population,sortlis):
        bestPopu = []
        for i in sortlis:
            bestPopu.append(population[int(i)])
        mu = np.mean(bestPopu,axis=0)
        sigma = np.std(bestPopu,axis=0)
        return mu,sigma
    # 生成新个体
    def generatePopu(self,mu,sigma,N):
        matri = np.zeros((len(mu),N))
        for idx,param in enumerate(zip(mu, sigma)):
            matri[idx] += param[1] * np.random.randn(N) + param[0]
        popu = matri.T
        return popu
    # 交叉
    def cross(self,Pc,population,bestIndiv):
        oldPopu = population
        newPopu = copy.deepcopy(population)
        N =len(population)
        n =len(population[0]) # 维度
        selN = int(N * Pc)
        selIdx = random.sample(range(N),selN)
        a = random.random()
        crossPoint = int(a*n)
        for idx in selIdx:
            newPopu[idx] = np.concatenate((bestIndiv[:crossPoint],oldPopu[idx][crossPoint:]),axis=0)
        return newPopu


# 可视化结果
def draw(answer1,answer2,Fnum,iternOld,iternNew):
    fig, ax = plt.subplots(1, Fnum)
    for i in range(Fnum):
        ax[i].set_title("Fun"+str(i+1)+"--iterOld"+str(iternOld[i])+"--iterNew"+str(iternNew[i]))
        ax[i].plot(answer1,'g-', label='normal')
        ax[i].plot(answer2,'b--',label='improved normal')
        # 创建图例 loc='best'自适应位置
        ax[i].legend(loc='best')
    plt.show()
def main():
    oldEDA = EDANormal()
    newEDA = EDANormal()
    answerOld,answerNew,iternOld,iternNew=[],[],[],[]
    for i in range(3):
        answer1, itern1 = oldEDA.runEDA(i+1,False)
        answer2, itern2 = newEDA.runEDA(i+1,True)
        answerOld.append(answer1)
        answerNew.append(answer2)
        iternOld.append(itern1)
        iternNew.append(itern2)
        print(i+1)
    draw(answer1, answer2,3,iternOld,iternNew)


if __name__=="__main__":
    main()
