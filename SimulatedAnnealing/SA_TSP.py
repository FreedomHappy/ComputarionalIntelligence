# encoding: utf-8
"""
@author: lin
@file: SA_TSP.py
@time: 2018/11/29 16:32
@desc:SA解决TSP问题,paper《旅行商问题的一种模拟退火算法求解》
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import math
"""
1.初始化
2.降温循环{
（1）热平衡循环{
        1）产生一个领域解-领域函数
        2）更新解-接收概率
        3）结束-内层平衡
    }
      
（2）降温冷却-冷却控制
}
3.结束
"""

# 加载数据
def dataLoad():
    dataSet = []
    yData = []
    xData = []
    with open('data.txt') as f:
        lines = f.readlines()
        for line in lines:
            str = line.strip('\n').split(' ')
            xData.append(float(str[0]))
            yData.append(float(str[1]))
            dataSet.append((float(str[0]), float(str[1])))
    return xData, yData, dataSet
# 计算向量距离
def cal_vector(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))
# 获取城市距离矩阵
def get_cdis(cities):
    distances = np.zeros((len(cities), len(cities)))
    for i, this in enumerate(cities):
        line = []
        for j, next in enumerate(cities):
            if i == j:  line.append(0.0)
            else: line.append(cal_vector(this, next))
        distances[i] = line
    return distances

class SA_TSP:
    def __init__(self,distances):
        self.distances = distances
        self.N = len(self.distances) # 城市个数
        self.T0 = 30  # 初始温度
        self.alpha = 0.97 # 温度衰减系数
        self.L = 30 # 热平衡探索迭代次数
        self.K = 15 # 波尔兹曼常数
        self.s0 = random.sample(range(self.N),self.N) # 初始状态即初始解
        self.overT = 0.01 # 结束温度
    # 运行
    def runSA(self):
        # 初始温度为self.T0，结束温度默认为0
        S = copy.deepcopy(self.s0)
        # Ts = range(self.T0+1)[::-1]
        T = self.T0
        iter_num = 0
        answer = []
        while True:
            iter_num += 1
            S,T,change = self.annealing(S,T,iter_num)
            answer.append(S)
            # print("第{}次迭代，当前温度{}，路径长度为:{},解为:\n{}".format(iter_num,T,self.objectFun(S),S))
            if not change and T<self.overT:
                break
        return answer
    # 冷却
    def annealing(self,S,T,iter_num):
        for idx in range(self.L):
            S,change = self.thermalEquilibrium(S,T)
        # 缓慢降温
        # T = self.T0/math.log10(float(1+iter_num))
        # 快速降温
        # T= self.T0/(1+iter_num)
        # 论文提供的降温方法
        T = T*self.alpha
        return S,T,change
    # 热平衡探索
    def thermalEquilibrium(self,S,T):
        newSolu = self.createNewSolu(S)
        delta = self.objectFun(newSolu)-self.objectFun(S)
        # Metropolis 准则
        if delta<0:
            return newSolu,True
        else:
            X = -(delta / (self.K*T))
            proba = math.exp(X)
            proLis = [1-proba,proba]
            idx = self.RWS(proLis) # 采用轮盘赌，返回0则不接受；返回1则接受
            if idx:
                return newSolu,True
            else:
                return S,False
    # 轮盘赌算法
    def RWS(self,proLis):
        sumProba = 0
        r = np.random.uniform()
        for i in range(len(proLis)):
            sumProba += proLis[i]
            if r < sumProba:
                return i

    # 产生新解,二变换法
    def createNewSolu(self,S):
        loc = random.sample(range(0, self.N), 2)
        loc.sort(reverse=False)
        u =  S[loc[0]]
        uAfter = S[loc[0]+1]
        v = S[loc[1]]
        vBefor = S[loc[1]-1]
        # 开始变换
        newSolu = copy.deepcopy(S)
        newSolu[loc[0]] = v
        newSolu[loc[0]+1] = vBefor
        newSolu[loc[1]] = u
        newSolu[loc[1]-1] = uAfter
        return newSolu
    # 目标函数,计算总路径长度
    def objectFun(self,path):
        distances = self.distances
        sum = 0
        for this, next in zip(path[:-1], path[1:]):
            sum += distances[this][next]
        return sum

# 可视化结果
def draw( answer,bestLen):
    # # 绘制城市散点图
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.set_title('cities')
    # ax1.scatter(X, Y)

    # # 绘制哈密顿回路
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.set_title("route")
    # path = np.concatenate((path, path[0:1]))
    # print('path', path)
    # for from_, to_ in zip(path[:-1], path[1:]):
    #     from_ = int(from_)
    #     to_ = int(to_)
    #     ax2.plot((data_cities[from_][0], data_cities[to_][0]),
    #             (data_cities[from_][1], data_cities[to_][1]), 'ro-')

    # 绘制次优解求解趋势图
    ans = []
    for i, j in zip(answer, range(len(answer))):
        ans.append([i, j])
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title('bestLen:{}'.format(bestLen))
    ax3.plot(answer, 'r-')
    plt.show()


def main():
    xd,yd,dataSet = dataLoad()
    dis = get_cdis(dataSet)
    sa = SA_TSP(dis)
    answer = sa.runSA()
    answer = list(map(sa.objectFun,answer))
    draw(answer,min(answer))
if __name__=="__main__":
    main()