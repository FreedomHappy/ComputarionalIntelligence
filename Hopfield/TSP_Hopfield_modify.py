# encoding: utf-8
"""
@author: lin
@file: TSP_Hopfield_modify.py
@time: 2018/10/13 16:31
@desc: "
"""

"""使用优化过的能量函数和状态方程"""
import numpy as np
from matplotlib import pyplot as plt

class TSP_Hopfield(object):
    def __init__(self, cities):
        self.cities = cities
        N = len(cities)
        self.N = N
        self.distance = self.get_distance()
        self.A = N * N
        self.D = N/2
        self.U0 = 0.0009
        self.step = 0.0001
        self.num_iter = 10000
        # 初始化输入输出状态
        self.U = 0.5 * self.U0 * np.log10(N-1)\
                + (2 * (np.random.random((N, N))) - 1)
        self.V = self.update_V()
        self.dU = None

    # 根据坐标获取两个城市的距离
    def vec_dis(self, vec1, vec2):
        return np.linalg.norm(np.array(vec1) - np.array(vec2))

    # 获取两两城市的距离矩阵
    def get_distance(self):
        distance = np.zeros((self.N, self.N))
        for i, curr_city in enumerate(self.cities):
            line = []
            [line.append(self.vec_dis(curr_city, other_city))
             if i != j else line.append(0.0)
             for j, other_city in enumerate(self.cities)]
            distance[i] = line
        return distance

    # 更新输入状态
    def update_U(self):
        return self.U + self.dU * self.step

    # 更新输出状态
    def update_V(self):
        return 0.5 * (1 + np.tanh(self.U / self.U0))

    # 动态方程
    def diff_U(self):
        row_sum = np.sum(self.V, axis=1)-1
        col_sum = np.sum(self.V, axis=0)-1
        f1 = np.zeros((self.N, self.N))
        f2 = np.zeros((self.N, self.N))
        for i in range(self.N):
            f1[i] += row_sum[i]
        for i in range(self.N):
            f2[i] += col_sum
        # 将第0列放到最后一列
        # c0 = self.V[:, 0] 切割出来数组只有一维
        c = np.concatenate((self.V[:, 1:], self.V[:, 0:1]), axis=1)
        f3 = np.dot(self.distance, c)
        return -self.A * (f1 + f2) - self.D * f3

    # 能量函数计算
    def get_energy(self):
        # 获取子式一
        f1 = np.sum(np.power(np.sum(self.V, axis=0) - 1, 2))
        # 获取子式二
        f2 = np.sum(np.power(np.sum(self.V, axis=1) - 1, 2))
        # 获取子式三
        idx = list(range(1, self.N)) + [0]
        # f3 = self.distance * self.V[:, idx]
        # f3 = np.sum(np.multiply(self.V, f3))
        V1 = self.V[:, idx]
        f3 = 0.0  # type: float
        for x in range(self.N):
            for y in range(self.N):
                f3 = f3 + self.distance[x][y] * (np.sum(V1[x] * V1[y]))
        return 0.5 * (self.A * (f1 + f2) + self.D * f3)

    def create_path(self):
        newV = np.zeros([self.N, self.N])
        route = []
        for i in range(self.N):
            # 找寻这一列里的最大值，即选定这i次访问的城市
            mm = np.max(self.V[:, i])
            for j in range(self.N):
                # 创建newV 添加城市x序号到route
                if self.V[j, i] == mm:
                    newV[j, i] = 1
                    route += [j]
                    break
        return route, newV

    def route_len(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.distance[path[i]][path[i+1]]
        return dis

    # 可视化哈密顿回路和能量趋势
    def draw_H_E(self, H_path, energys, best_dis, best_iter):
        fig, ax = plt.subplots(1, 2)
        # 绘制哈密顿回路
        ax[0].set_xlim(0, 7)
        ax[0].set_ylim(0, 7)
        for (from_, to_) in H_path:
            ax[0].plot((self.cities[from_][0], self.cities[to_][0]),  (self.cities[from_][1], self.cities[to_][1]), 'bo-')
            ax[0].annotate(s=chr(97 + to_), xy=self.cities[to_], xytext=(-8, 8), textcoords='offset pixels', fontsize=20)
        ax[0].grid()
        ax[0].set_title("route_len:{}".format(best_dis))
        # 绘制能量趋势图
        ax[1].plot(np.arange(0, len(energys)), energys, 'b')
        ax[1].set_ylabel("energy")
        ax[1].set_xlabel("iter")
        ax[1].set_title("energy:{}".format(energys[best_iter]))
        # 绘图显示
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

    def TSP_search(self):
        # 每次迭代的能量
        energys = np.array([0.0 for x in range(self.num_iter)])
        # 最优距离
        best_distance = np.inf
        # 最优路线
        best_route = []
        # 哈密顿回路
        H_path = []
        # 次优解的迭代次序
        best_iter = 0

        # 开始迭代训练网络
        for n in range(self.num_iter):
            # 计算dU
            self.dU = self.diff_U()
            # 更新输入状态
            self.U = self.update_U()
            # 更新输出状态
            self.V = self.update_V()
            # 计算当前网络的能量
            energys[n] = self.get_energy()
            # 生成合法路径
            route, newV = self.create_path()
            # 如果route路径合法
            if len(np.unique(route)) == self.N:
                route.append(route[0])
                dis = self.route_len(route)
                # 如果本次route路径变短
                # if dis < best_distance:
                if energys[n] < energys[best_iter]:
                    H_path = []
                    best_distance = dis
                    best_route = route
                    best_iter = n
                    [H_path.append((route[i], route[i+1])) for i in range(len(route)-1)]
                    print("第{}次迭代找到的次优解距离为：{}，能量为：{}，路径为：".format(n, best_distance, energys[n]))
                    [print(chr(97+v), end=',' if i < len(best_route)-1 else '\n')for i, v in enumerate(best_route)]
        if len(H_path) > 0:
            self.draw_H_E(H_path, energys, best_distance, best_iter)
        else:
            print('没有找到最优解')
def main():
    cities = np.array([[2, 6], [2, 4], [1, 3], [4, 6], [5, 5], [4, 4], [6, 4], [3, 2]])
    tsp = TSP_Hopfield(cities)
    tsp.TSP_search()

if __name__  ==  '__main__':
    main()