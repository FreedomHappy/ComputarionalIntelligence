# encoding: utf-8
"""
@author: lin
@file: PSO_TSP.py
@time: 2018/11/15 17:28
@desc:paper《粒子群优化算法求解旅行商问题》
    论文的创新之处在于提出了：交换子和交换序
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
import copy
# 加载数据
def dataLoad():
    dataSet = []
    yData = []
    xData = []
    with open('cities1.txt') as f:
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


"""粒子类"""
class Particle_TSP:
    def __init__(self,dimensions):# 此处dimensions即为城市数
        self.dimension = dimensions
        # 初始化基本交换序
        self.Vn = int(1*dimensions)
        self.V = np.random.randint(0,dimensions,(self.Vn,2))
        # 初始化城市路径
        self.X = np.array(random.sample(range(dimensions),dimensions))

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
class ParticleSwarm_TSP:
    def __init__(self,distances):
        self.distances = distances
        self.N = 300 # 粒子数
        self.dimensions = len(distances[0]) # X,V 维度
        self.iter_num = 300 # 迭代次数
        # 初始化群体各粒子
        self.ps = []
        result = []
        for i in range(self.N):
            # 初始化粒子的X,V
            p = Particle_TSP(self.dimensions)
            # 初始化 粒子最优pBest
            r = self.fitness(p.getX())
            p.setpBest(p.getX())
            self.ps.append(p)
            result.append(r)
        # 全局最优
        self.gBest = self.ps[int(np.argmin(result))].getX()

    def runPSO_TSP(self):
        self.update()
        iter_num = self.iter_num
        answer = []
        for iter in range(iter_num):
            answer.append(self.update())
            if iter%10==0:
                print("第{}次迭代：".format(iter+1))
                print('gBest',self.fitness(self.gBest))
        return answer,self.gBest,self.fitness(self.gBest)

    # 更新粒子群
    def update(self):
        ps = self.ps
        pathLenLis = []
        for p in ps:
            pathLenLis.append(self.updateParticle(p))
        min = np.min(pathLenLis)
        gBestLen = self.fitness(self.gBest)
        if min<gBestLen:
            self.gBest = copy.deepcopy(ps[int(np.argmin(pathLenLis))].getX())
        return min
    # 更新粒子V，X
    def updateParticle(self,p):
        # 更新基本交换序
        alpha = random.random()
        beta = random.random()
        # print('a',alpha,'b',beta)
        A = self.getSwapSequence(p.getpBest(),p.getX())
        B = self.getSwapSequence(self.gBest,p.getX())
        # print('A',A,'B',B)
        A = random.sample(A,int(alpha*len(A)))
        B = random.sample(B,int(beta*len(B)))
        V = np.array(list(p.getV())+A+B)
        # print('A', A, 'B', B)
        # print('V',V)
        # 更新位置X
        X = p.getX()
        for v in V:
            temp = X[v[0]]
            X[v[0]] = X[v[1]]
            X[v[1]] = temp
        p.setV(V)
        p.setX(X)
        # 更新pBest
        pathLen = self.fitness(p.getX())
        pBestLen = self.fitness(p.getpBest())
        if pathLen<pBestLen:
            p.setpBest(copy.deepcopy(p.getX()))
        return pathLen
    # 适应度函数
    def fitness(self, path):
        distances = self.distances
        sum = 0
        for this, next in zip(path[:-1],path[1:]):
            sum += distances[this][next]
        return sum

    # 获取基本交换序列
    def getSwapSequence(self, P, X_copy):
        swapLis = []
        X = copy.deepcopy(X_copy)
        for Pidx, p in enumerate(P):
            if (P == X).all():
                break
            Xidx = list(X).index(p)
            swapLis.append((Pidx, Xidx))
            temp = X[Pidx]
            X[Pidx] = X[Xidx]
            X[Xidx] = temp
        return swapLis
# 可视化结果
def draw(X, Y, data_cities, answer, path,bestLen):
    # 绘制城市散点图
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.set_title('cities')
    # ax1.scatter(X, Y)

    # 绘制哈密顿回路
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("route")
    path = np.concatenate((path, path[0:1]))
    print('path', path)
    for from_, to_ in zip(path[:-1], path[1:]):
        from_ = int(from_)
        to_ = int(to_)
        ax2.plot((data_cities[from_][0], data_cities[to_][0]),
                (data_cities[from_][1], data_cities[to_][1]), 'ro-')

    # 绘制次优解求解趋势图
    ans = []
    for i, j in zip(answer, range(len(answer))):
        ans.append([i, j])
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title('bestLen:{}'.format(bestLen))
    ax3.plot(answer, 'r-')

    plt.show()
def PSO_TSP():
    xd, yd, dataSet = dataLoad()
    distances = get_cdis(dataSet)
    PS = ParticleSwarm_TSP(distances)
    answer,path,bestLen=PS.runPSO_TSP()
    draw(xd,yd,dataSet,answer,path,bestLen)

if __name__=="__main__":
    PSO_TSP()