# encoding: utf-8
"""
@author: lin
@file: AS_TSP.py
@time: 2018/11/8 16:42
@desc:《求解旅行商问题的Matlab 蚁群仿真研究》
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
"""
AS:蚂蚁系统；ACS：蚁群系统
"""
"""
蚁群优化算法基本流程
AS：蚂蚁系统
一、路径构建
1.初始化
    n个城市，m只蚂蚁
    贪心算法得到路径
    求得初始Tau0
    初始化所有边上的信息素Tau_ij
2.为每只蚂蚁随机选择出发城市
3.为每只蚂蚁选择下一访问城市
    以蚂蚁1为例，当前城市i=A,可访问城市集合J1_i={B,C,D}
    启发式信息*信息素浓度，alpha和beta控制权重关系，
    一般alpha=1,beta=2～5
    B:(Tau_AB**alpha)*(Eta_AB**beta)
    C:...
    D:...
    p(B)=B/(B+C+D)
    p(C)=..
    p(D)=..
    用轮盘赌算法选择下一访问城市
4.重复 步骤3.直至全部路径构造完毕
二、信息素更新，采用ant-cycle
1.计算每只蚂蚁构造的路径长度Ck
2.更新每条边上的信息素
    当前边的信息素蒸发+所有蚂蚁在当前边的信息素释放
    （1） 信息素蒸发rho,一般rho = 0.5
    （2）信息素释放，delta_Tauk_ij = 1/Ck
三、结束条件判断
1.满足输出最优解
2.否则转向步骤一.2.
"""
# 加载数据
def dataLoad():
    dataSet = []
    yData = []
    xData = []
    with open('cities1.txt') as f:
        lines = f.readlines()
        for line in lines:
            str = line.strip('\n').split(' ')
            xData.append(int(str[0]))
            yData.append(int(str[1]))
            dataSet.append((int(str[0]), int(str[1])))
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
# 贪心算法计算TSP路径
def greedy_TSP(distances):
    n = len(distances[0])
    depar_city = random.sample(range(0, n), 1)[0] # 返回是一个数组所以要索引
    S = [depar_city]
    sumpath = 0
    while True:
        min = np.inf
        for i, dis in enumerate(distances[S[-1]]):
            if (i in S) or (dis==0) :
                continue
            if dis<min:
                min = dis
                idx = i
        S.append(idx)
        sumpath += min
        if len(S) == n:
            break
    sumpath += distances[S[-1]][depar_city]
    S.append(depar_city)
    return  S, sumpath


# 可视化结果
def draw(X, Y, data_cities, answer, path,bestLen):
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

# 蚁群算法类
class ACO_TSP:
    def __init__(self,xd,yd,dataSet):
        # n个城市，m只蚂蚁
        self.n = len(xd)
        self.m = 50
        # 参数初始化
        self.alpha = 1 # 启发式因子 AS
        self.beta = 6  # 期望启发因子 AS
        self.rho = 0.15 # 信息素挥发因子 AS
        self.Q = 100 # 信息素强度系数 快速收敛作用
        self.iter_num = 300 # 迭代次数
        self.q0 = 0.1 # 开发与探索平衡系数 ACS
        self.q0Change_iter = 50
        self.q0Change = 0.01
        self.xi = 0.5 # 信息素局部挥发速率 0<xi<1 ACS
        # 获取城市距离矩阵
        self.distances = get_cdis(dataSet)
        # 贪心算法得到初始路径
        S, sumpath = greedy_TSP(self.distances)
        self.greedyLen = sumpath
        # 求得初始Tau0
        # 初始化初始路径边上的信息素Tau_ij
        # ASTau0初始规则：Tau0 = self.m / sumpath
        # ACSTau0初始规则：
        Tau0 = 1 / (self.n*sumpath)
        self.Tau0 = Tau0
        self.Tau = np.zeros((self.n, self.n))  # Tau矩阵
        self.Tau[:,:] = Tau0
    def runACS(self):
        # 迭代次数iter_num
        iter_num = self.iter_num
        n = self.n
        distances = self.distances
        Tau = self.Tau
        # 最优路径
        # bestpath[0]存储路径长度，bestpath[1]存储路径序列
        bestpath = [[np.inf],[]]
        # answer 每次迭代的最小值
        answer = []
        for i in range(iter_num):
            # 一、路径构建
            # 2.为每只蚂蚁随机选择出发城市
            depar_city = []
            routes = []
            for j in range(n):
                city = random.sample((range(n)), 1)[0]
                depar_city.append(city)
                routes.append([city])

            # 3.为每只蚂蚁选择下一访问城市
            # 以蚂蚁1为例，当前城市i=A,可访问城市集合J1_i={B,C,D}
            # 构造未访问城市矩阵J
            J = []
            for k in depar_city:
                lis = list(range(n))
                lis.remove(k)
                J.append(lis)
            # 构造路径
            while True:
                routes, J = self.addCity(n, Tau, distances, routes, J,i)
                if len(routes[0]) == n:
                    for route in routes:
                        route.append(route[0])
                    break

            # 二、信息素更新，采用ant-cycle
            # 1.计算每只蚂蚁构造的路径长度Ck
            C = []
            for route in routes:
                C.append(self.cal_routeLen(route, distances))
            # 2.更新每条边上的信息素
            Tau = self.updateTau_ACS(Tau, routes, C)
            if i%10==0:
                print('迭代次数{}'.format(i + 1))
                print(C)
            # 遴选最优路径
            minLen = min(C)
            answer.append(minLen)
            if minLen<bestpath[0]:
                bestpath[0]=minLen
                bestpath[1]=routes[C.index(minLen)]

        print("greedyLen:{}".format(self.greedyLen))
        print('最短路径长:{}\n路径为:{}'.format(bestpath[0],bestpath[1]))
        return bestpath,answer
    # 添加下一个访问城市
    def addCity(self,n,Tau,distances,routes,J,iter_num):
        # 启发式信息*信息素浓度，alpha和beta控制权重关系，
        # 一般alpha=1,beta=2～5
        # 用轮盘赌算法选择下一访问城市
        alpha = self.alpha
        beta = self.beta
        q0 = self.q0 # ACS系数,伪随机比例规则
        # 到了指定迭代期 改动q0
        if iter_num==self.q0Change_iter:
            q0 = self.q0Change
        xi = self.xi # ACS系数，信息素局部挥发速率
        Tau0 = self.Tau0 #信息素局部更新规则参数
        for i in range(n):
            # 蚂蚁i访问城市选择
            this = routes[i][-1]
            line = []
            for next in J[i]:
                # 启发式信息 * 信息素浓度
                # 例如城市B:(Tau_AB**alpha)*(Eta_AB**beta)
                # 只用 Tau矩阵右上角
                if this>next:
                    value = (Tau[next][this] ** alpha) * ((1 / distances[this][next]) ** beta)
                else:
                    value = (Tau[this][next] ** alpha) * ((1 / distances[this][next]) ** beta)
                line.append(value)
            line = np.array(line)
            # p(B)=B/(B+C+D)
            line = line / np.sum(line)
            # 状态转移规则，ACS规则
            Roulette_data = {}
            for kv in zip(J[i], line):
                Roulette_data[kv[0]] = kv[1]
            roulette = sorted(Roulette_data.items(), key=lambda e: e[1])
            q = random.random()
            if q<=q0:
                # 开发
                nextcity = roulette[-1][0]
            else:
                # 探索，进入轮盘赌
                sump = 0
                for idx in range(len(roulette)):
                    sump += roulette[idx][1]
                    if q <= sump:
                        nextcity = roulette[idx][0]
                        break
            J[i].remove(nextcity)
            routes[i].append(nextcity)
            # 信息素局部更新规则
            this,next= routes[i][-2],routes[i][-1]
            if this>next:
                Tau[next][this] = (1 - xi) * Tau[next][this] + xi * Tau0
            else:
                Tau[this][next]=(1-xi)*Tau[this][next]+xi*Tau0
        return routes,J
    # 计算每个蚂蚁的路径长度
    def cal_routeLen(self,route, distances):
        routeLen = 0
        for this, next in zip(route[:-1],route[1:]):
            routeLen +=distances[this][next]
        return routeLen
    # 更新信息系素
    def updateTau(self,Tau, routes, C):
        # 当前边的信息素蒸发+所有蚂蚁在当前边的信息素释放
        #（1）信息素蒸发rho,一般rho = 0.5
        #（2）信息素释放，delta_Tauk_ij = 1/Ck
        rho = self.rho
        Tau = (1 - rho) * Tau
        for i,route in enumerate(routes):
            for this,next in zip(route[:-1],route[1:]):
                # 只用Tau矩阵右上角
                if this>next:
                    Tau[next][this] += 1/C[i]
                else:
                    Tau[this][next] += 1/C[i]
        return Tau
    # ACS信息素全局更新规则
    def updateTau_ACS(self,Tau, routes, C):
        # 当前边的信息素蒸发+所有蚂蚁在当前边的信息素释放
        # （1）信息素蒸发rho,一般rho = 0.5
        # （2）信息素释放，delta_Tauk_ij = 1/Ck
        rho = self.rho
        Q = self.Q
        Tau = (1 - rho) * Tau
        # 遴选最优路径
        minLen = min(C)
        minIdx = C.index(minLen)
        bestRoute = routes[minIdx]
        for this, next in zip(bestRoute[:-1], bestRoute[1:]):
            # 只用Tau矩阵右上角
            if this > next:
                Tau[next][this] += rho*(Q / C[minIdx])
            else:
                Tau[this][next] += rho*(Q / C[minIdx])
        return Tau

# 蚁群优化算法基本流程
# ACO：蚁群算法
def ACO_test():
    # 数据加载
    xd, yd, dataSet = dataLoad()
    aco = ACO_TSP(xd, yd, dataSet)
    bestpath,answer = aco.runACS()
    draw(xd,yd,dataSet,answer,bestpath[1],bestpath[0])

if __name__=="__main__":
    ACO_test()