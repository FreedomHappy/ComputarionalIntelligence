# encoding: utf-8
"""
@author: lin
@file: AS_TSP.py
@time: 2018/11/8 16:42
@desc:《求解旅行商问题的Matlab 蚁群仿真研究》
"""
import random
import numpy as np
import matplotlib as plt
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
# 产生各个蚂蚁的路径
def addCity(n,Tau,distances,routes,J):
    # 启发式信息*信息素浓度，alpha和beta控制权重关系，
    # 一般alpha=1,beta=2～5
    # 用轮盘赌算法选择下一访问城市
    alpha = 1
    beta = 2
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
        # 轮盘赌算法
        Roulette_data = {}
        for kv in zip(J[i], line):
            Roulette_data[kv[0]] = kv[1]
        roulette = sorted(Roulette_data.items(), key=lambda e: e[1])
        q = random.random()
        sump = 0
        for idx in range(len(roulette)):
            sump += roulette[idx][1]
            if q <= sump:
                nextcity = roulette[idx][0]
                break
        J[i].remove(nextcity)
        routes[i].append(nextcity)
    return routes,J
# 计算每个蚂蚁的路径长度
def cal_routeLen(route, distances):
    routeLen = 0
    for this, next in zip(route[:-1],route[1:]):
        routeLen +=distances[this][next]
    return routeLen
# 更新信息系素
def updateTau(Tau, routes, C):
    # 当前边的信息素蒸发+所有蚂蚁在当前边的信息素释放
    #（1）信息素蒸发rho,一般rho = 0.5
    #（2）信息素释放，delta_Tauk_ij = 1/Ck
    rho = 0.9
    Tau = (1 - rho) * Tau
    for i,route in enumerate(routes):
        for this,next in zip(route[:-1],route[1:]):
            # 只用Tau矩阵右上角
            if this>next:
                Tau[next][this] += 1/C[i]
            else:
                Tau[this][next] += 1/C[i]
    return Tau
# 蚁群优化算法基本流程
# AS：蚂蚁系统
def AS():
    # 一、路径构建
    # 1.初始化
    # n个城市，m只蚂蚁
    xd, yd, dataSet = dataLoad()
    n = len(xd) # 城市个数
    m = 31 # 蚂蚁个数
    distances = get_cdis(dataSet) # 获取城市距离矩阵
    
    # 贪心算法得到路径
    S, sumpath = greedy_TSP(distances)
    # 求得初始Tau0
    # 初始化所有边上的信息素Tau_ij
    Tau0 = m/sumpath
    Tau = np.zeros((n, n)) # Tau矩阵
    Tau[:,:] = Tau0 # 切片视图广播

    # 迭代次数iter_num
    iter_num = 1000
    for i in range(iter_num):
        print('迭代次数{}'.format(i+1))
        
        # 2.为每只蚂蚁随机选择出发城市
        depar_city = []
        routes = []
        for i in range(n):
            city = random.sample((range(n)), 1)[0]
            depar_city.append(city)
            routes.append([city])
            
        # 3.为每只蚂蚁选择下一访问城市
        # 以蚂蚁1为例，当前城市i=A,可访问城市集合J1_i={B,C,D}
        # 构造未访问城市矩阵J
        J = []
        for i in depar_city:
            lis = list(range(n))
            lis.remove(i)
            J.append(lis)
        # 构造路径
        while True:
            routes,J = addCity(n,Tau,distances,routes,J)
            if len(routes[0])==n:
                for route in routes:
                    route.append(route[0])
                break

        # 二、信息素更新，采用ant-cycle
        # 1.计算每只蚂蚁构造的路径长度Ck
        C = []
        for route in routes:
            C.append(cal_routeLen(route,distances))
        # 2.更新每条边上的信息素
        Tau = updateTau(Tau,routes,C)
        print(C)

if __name__=="__main__":
    AS()