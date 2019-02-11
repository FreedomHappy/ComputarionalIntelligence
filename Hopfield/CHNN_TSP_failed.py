# encoding: utf-8
"""
@author: lin
@file: CHNN_TSP_failed.py
@time: 2018/10/2 19:01
@desc:
"""

"""使用最原始的TSP能量函数和TSP状态方程，十分的麻烦"""
import numpy as np

distance = np.array([[0, 7, 6, 10, 13],
                     [7, 0, 7, 10, 10],
                     [6, 7, 0, 5, 9],
                     [10, 10, 5, 0, 6],
                     [13, 10, 9, 6, 0]
                     ])
distance2 = np.array([[],[],[]])

class CHNN_TSP(object):
    def __init__(self, num_cities):
        self.distance = np.zeros((num_cities, num_cities))#城市距离矩阵
        self.weights = np.zeros((num_cities, num_cities))#权重矩阵
        self.v = np.zeros((num_cities, num_cities))#状态矩阵
        self.u = np.zeros((num_cities, num_cities))
        self.num_c = num_cities#城市数
        self.num_n =num_cities*num_cities#神经元个数

        #能量函数参数
        self.A = 1
        self.B = 1
        self.C = 7.85
        self.D = 2.2
        self.u0 = 0.02
        self.energy = 10000

        #选定神经元的位置索引
        self.vi=0
        self.vj=0

        self.distance = distance
    def energy_func(self):
        first = 0
        for x in range(self.num_c):
            for i in range(self.num_c)[:-1]:
                for j in range(self.num_c)[i+1:]:
                    first = first + self.v[x][i]*self.v[x][j]
        second = 0
        for y in range(self.num_c):
            for i in range(self.num_c)[:-1]:
                for j in range(self.num_c)[i+1:]:
                    second = second + self.v[i][y]*self.v[j][y]
        third = 0
        for i in range(self.num_c):
            for j in range(self.num_c):
                third = third + self.v[i][j]
        fourth = 0
        for x in range(self.num_c):
            for y in range(self.num_c):
                if x == y: continue
                for i in range(self.num_c):
                    if i-1 < 0: a=0
                    else:   a = self.v[y][i-1]
                    if i+1 >= self.num_c: c=0
                    else:   c = self.v[y][i+1]
                    fourth = fourth + self.distance[x][y]*self.v[x][i]*(a + c)
        energy = (self.A/2)*first + (self.B/2)*second + (self.C/2)*(third-self.num_c)**2 + (self.D/2)*fourth
        return energy

    def init_u(self):
        u00 = self.u0*np.log10(self.num_n-1)/2
        sigma = 2*np.random.random_sample([self.num_c,self.num_c])-1
        self.u = u00 + sigma
    def update(self):
        delta_u = np.zeros((self.num_c,self.num_c))
        T = 1
        for x in range(self.num_c):
            for i in range(self.num_c):
                fourth = 0
                for y in range(self.num_c):
                    if i-1<0: a=0
                    else: a=self.v[y][i-1]
                    if i+1>0: c=0
                    else: c=self.v[y][i+1]
                    fourth = fourth + self.distance[x][y]*(a+c)
                delta_u[x][i] = -self.u[x][i]/T - self.A*(np.sum(self.v[x])-self.v[x][i])\
                                -self.B*(np.sum(self.v[:,i])-self.v[x][i])\
                                -self.C*(np.sum(self.v)-self.num_c)\
                                -self.D*fourth
        lamda = 1
        self.u = self.u + lamda*delta_u
        print(self.u)

        #更新位置参数
        # a, b = np.random.randint(0, self.num_c-1, 2)
        a=self.vi
        b=self.vj
        self.vi=(self.vi+1)%self.num_c
        temp = self.v[a][b]
        self.v[a][b] = (1+np.tanh(self.u[a][b]/self.u0))/2
        # if self.energy_func()>self.energy:
        #     self.v[a][b] = temp
        # else: self.energy = self.energy_func()
        print(self.v)

    def preditc(self, epochs):
        self.init_u()
        for i in range(epochs):
            #更新位置参数
            if i+1%self.num_c==0:
               self.vj=(self.vj+1)%self.num_c
            print(i)
            self.update()

        print(self.v)

    def init_weights(self):
        sigma = []
        for i in range(self.num_c):
            for j in range(self.num_c):
                sig = []
                if i==j: sig.append(1)
                else : sig.append(0)
            sigma.append(sig)
        pass


def main():
    tsp = CHNN_TSP(5)
    tsp.preditc(100)
    pass


if __name__ == "__main__":
    main()
