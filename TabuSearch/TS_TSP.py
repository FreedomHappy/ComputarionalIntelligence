# encoding: utf-8
"""
@author: lin
@file: TS_TSP.py
@time: 2018/12/6 16:21
@desc:
"""
import math

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
# 加载数据
def dataLoad():
    dataSet = []
    yData = []
    xData = []
    with open('cities2.txt') as f:
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

class TS_TSP:
    def __init__(self,distances):
        # 禁忌对象定义为交换城市
        # 解的领域映射为2-opt
        self.distances = distances
        self.cities_num = len(distances)
        self.tabuLength = 5
        self.iter_out = 100
        # 特赦准则。本文的特赦规则定义为: 当当前最优解未下
        # 降的次数超过给定值时, 则特赦禁忌表中的最优解,
        # 将其作为下一次迭代的初始解;
        self.ACnum = self.tabuLength+5
        self.MaxAC = 10 # 特赦准则，最大特赦次数
        pass
    def run(self):
        thePath = random.sample(range(self.cities_num),self.cities_num)
        bestLen = self.fitness(thePath)
        tabu = [] # 禁忌表
        answer = [bestLen] # 解集
        iter_num = 0
        bestPath = thePath
        ACCount = 0 # 计算特赦准则中最优解未下降次数超过的定值
        while 1:
            iter_num += 1
            # 迭代次数限制
            if iter_num>self.iter_out:
                break
            # 更新解
            minPath, minLen, theT =self.update(copy.deepcopy(thePath),copy.deepcopy(tabu))
            answer.append(minLen)
            thePath = minPath
            # 更新最优解
            if minLen<bestLen:
                bestLen = minLen
                bestPath = minPath
                ACCount =0
            else:
                ACCount +=1
                if ACCount>self.ACnum:
                    self.MaxAC -=1
                    if self.MaxAC <0:
                        break
                    thePath = self.getBestFromTabu(tabu)
            print("第{}次迭代,minLen:{}\n , bestLen:{}\n thePath:{}".format \
                  (iter_num, minLen, bestLen,thePath))
            tabu = self.updateTabu(copy.deepcopy(tabu),theT)
            # print('tabu:',tabu)
        return answer,bestLen
    def updateTabu(self, tabu, theT):
        # for t in tabu:
        #     t[1] -= 1
        #     if t[1]==0:
        #         tabu.remove(t)
        i = 0
        while i < len(tabu):
            tabu[i][1] -= 1
            if  tabu[i][1]== 0:
                tabu.pop(i)
                i -= 1
            i += 1
        tabu.append([theT,self.tabuLength])
        return tabu

    def update(self, path, tabu):
        upPath = path[1:]
        solutions,T = self.createSoultions(copy.deepcopy(upPath),tabu,path[0:1])
        newPaths = []
        for s in solutions:
            newPaths.append(path[0:1]+s)
        pathLens = []
        for newPath in newPaths:
            pathLens.append(self.fitness(newPath))
            # print(self.fitness(newPath))
        # print(pathLens)
        minLen = min(pathLens)
        minIdx = pathLens.index(minLen)
        minPath = newPaths[minIdx]
        theT = T[minIdx] # 选出禁忌对
        return minPath,minLen,theT

    def createSoultions(self,upPath,tabu,p0):
        solutions = []
        T = []
        newTabu = []
        for t in tabu:
            newTabu.append(t[0])
        for i,first in enumerate(upPath):
            for j,second in enumerate(upPath[i+1:]):
                # 检查是否在禁忌表中
                # # 禁忌对象为交换城市
                # if (first,second) in newTabu or \
                #         (second,first) in newTabu:
                #     continue
                path = copy.deepcopy(upPath)
                path[i] = second
                path[(i + 1) + j] = first
                # # 禁忌对象为交换城市
                # T.append((first,second))
                # 禁忌对象为目标值
                if (p0+path) in newTabu:
                    continue
                solutions.append(path)
                T.append(p0+path)
        return solutions, T
    def getBestFromTabu(self,tabu):
        tabuPath = []
        pathLen = []
        for t in tabu:
            tabuPath.append(t[0])
        for path in tabuPath:
            pathLen.append(self.fitness(path))
        minIdx = pathLen.index(min(pathLen))
        return tabuPath[minIdx]
    def fitness(self,route):
        routeLen = 0
        for this, next in zip(route[:-1],route[1:]):
            routeLen +=self.distances[this][next]
        return routeLen

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
    xData, yData, dataSet = dataLoad()
    distances = get_cdis(dataSet)
    # print(distances)
    ts_tsp = TS_TSP(distances)
    ans, best = ts_tsp.run()
    draw(ans,best)
    pass

if __name__=="__main__":
    main()