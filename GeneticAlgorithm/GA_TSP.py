# encoding: utf-8
"""
@author: lin
@file: GA_TSP.py
@time: 2018/10/18 16:36
@desc:基于遗传算法的旅行商问题的研究
"""
import numpy as np
import random
import matplotlib.pyplot as plt
"""
一、对访问城市序列进行排列组合的方法编码
    保证了每个城市经过且只经过一次
二、生成初始种群
    个体数目（20,100）
三、计算适应度函数，即计算遍历所有城市的距离
四、选择算子--最优保存法
五、交叉算子--有序交叉
六、变异算子--倒置变异法
七、终止条件
    迭代次数（40,80）
"""

# 构建城市距离矩阵
def dataLoad():
    dataSet = []
    yData = []
    xData = []
    with open('X.txt') as X, open('Y.txt') as Y:
        xlines = X.readlines()
        ylines = Y.readlines()
        for xs, ys in zip(xlines, ylines):
            str1 = xs.strip('\n').split(' ')
            str2 = ys.strip('\n').split(' ')
            xd = []
            yd = []
            for word in str1:
                xd.append(float(word))
                xData.append(float(word))
            for word in str2:
                yd.append(float(word))
                yData.append(float(word))
            for xy in zip(xd, yd):
                dataSet.append(list(xy))
    return xData, yData, dataSet
def cal_vector(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))
def get_cdis(cities):
    distances = np.zeros((len(cities), len(cities)))
    for i, this in enumerate(cities):
        line = []
        for j, next in enumerate(cities):
            if i == j:  line.append(0.0)
            else: line.append(cal_vector(this, next))
        distances[i] = line
    return distances

# 目标函数
def objective_func(indiv, distances):
    sum = 0
    for from_, to_ in zip(indiv[:-1], indiv[1:]):
        sum += distances[int(from_)][int(to_)]
    return sum
# 适应度函数
def fitness_func(obj):
    return 1 / obj

# 选择算子
def selection(pop):
    max = {'idx': -1, 'fit': 0}
    min = {'idx': -1, 'fit': np.inf}
    for i, line in enumerate(pop):
        if line[-1] >= max['fit']:
            max['fit'] = line[-1]
            max['idx'] = i
        if line[-1] <= min['fit']:
            min['fit'] = line[-1]
            min['idx'] = i
    # 不进行交叉和变异运算，直接替换
    pop[min['idx']] = pop[max['idx']]
    print('\n', pop)
    return max['idx'], min['idx'], pop

# 检查是否符合TSP约束条件
def check(C11, Y1, Y2):
    Y1.reshape(-1)
    Y2.reshape(-1)
    for i, c in enumerate(C11):
        c1 = c
        while 1:
            if c1 in Y1:
                # bug 此处idx易出现二维数组
                idx = np.argwhere(Y1 == c1)
                # print('idx:', idx)
                # print('Y2[idx]:', Y2[idx])
                c1 = Y2[np.sum(idx)]
            else:
                C11[i] = c1
                break
    return C11

# 交配算子
def crossover(selec1, selec2, N, s, pop):
    Pc = (0.99 - 0.4) * np.random.random_sample()+0.4
    # Pc = 0.9
    print('交配概率：', Pc)
    mat_proba = np.random.rand(s)
    print('个体交配率：',mat_proba)
    mat_idx = []
    for i in range(s):
        if i == selec1 or i == selec2 : continue
        if mat_proba[i] < Pc: mat_idx.append(i)
    print('可交配序列：',mat_idx)
    # 若为奇数个，则单身
    half = len(mat_idx) // 2
    print('half:',half)
    np.random.shuffle(mat_idx)
    father = mat_idx[:half]
    np.random.shuffle(father)
    mother = mat_idx[half:]
    np.random.shuffle(mother)
    # 染色体交叉
    for i in range(half):
        print('第{}次染色体交叉'.format(i))
        # 取出父母索引
        idxf = father[i]
        idxm = mother[i]
        loc = random.sample(range(0, N), 2)
        loc.sort(reverse = False)
        # print('\n', 'popf', pop[idxf])
        # print('popm', pop[idxm])
        # Y1，Y2子串
        Y1 = pop[idxf][loc[0]:loc[1]]
        Y2 = pop[idxm][loc[0]:loc[1]]
        # print('Y1:', Y1)
        # print('Y2:', Y2)
        # 取出交叉段
        C11 = pop[idxm][:loc[0]]
        C12 = pop[idxm][loc[1]:-1]
        C21 = pop[idxf][:loc[0]]
        C22 = pop[idxf][loc[1]:-1]
        # print('C11:', C11)
        # print('C12:', C12)
        # print('C21:', C21)
        # print('C22:', C22)
        # 检查是否符合TSP约束
        C11 = check(C11, Y1, Y2)
        C12 = check(C12, Y1, Y2)
        C21 = check(C21, Y2, Y1)
        C22 = check(C22, Y2, Y1)
        # 染色体交叉
        C1 = np.concatenate((C11, Y1, C12, [pop[idxf][-1]]))
        C2 = np.concatenate((C21, Y2, C22, [pop[idxm][-1]]))
        # print('C1:', C1)
        # print('C2:', C2)
        pop[idxf] = C1
        pop[idxm] = C2
    return  pop

# 变异算子
def mutation(N, s, pop):
    Pm = (0.1 - 0.001) * np.random.random_sample() + 0.001
    # Pm = 0.2
    mut_prob = np.random.rand(s)
    mut_idx = []
    for i, p in enumerate(mut_prob):
        if p < Pm: mut_idx.append(i)

    for i in mut_idx:
        loc = random.sample(range(0, N), 2)
        loc.sort(reverse=False)
        # 倒置子串
        Y = pop[i][loc[0] + 1:loc[1]]
        Y = Y[::-1]
        pop[i] = np.concatenate((pop[i][:loc[0] + 1], Y, pop[i][loc[1]:]))
    # print('pop:', pop)
    return pop
# 可视化结果
def draw(X, Y, data_cities, answer, path):
    # # 绘制城市散点图
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.set_title('cities')
    # ax1.scatter(X, Y)
    
    # # 绘制哈密顿回路
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # path = np.concatenate((path, path[0:1]))
    # print('path', path)
    # for from_, to_ in zip(path[:-1], path[1:]):
    #     from_ = int(from_)
    #     to_ = int(to_)
    #     ax2.plot((data_cities[from_][0], data_cities[to_][0]),
    #             (data_cities[from_][1], data_cities[to_][1]), 'ro-')
    
    # 绘制次优解求解趋势图
    ans =[]
    for i, j in zip(answer, range(len(answer))):
       ans.append([i, j])
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title('last answer:{}'.format(answer[-1]))
    ax3.plot(answer, 'r-')

    plt.show()

def main():
    # 基本参数
    # 城市矩阵
    X, Y, data_cities = dataLoad()
    X = X[:30]
    Y = Y[:30]
    data_cities = data_cities[:30]
    N = len(data_cities)  # 城市数目
    s = 30  # 个体数目
    dis = get_cdis(data_cities)  # 城市距离矩阵
    iter_num = 1700

    #  二、初始群体设定
    # 构建s行t列的pop矩阵, s为个体数目
    # 该矩阵每一行的前30个元素表示经过的城市编号，
    # 最后一个表示适应度函数的取值
    t = N + 1
    pop = np.zeros((s, t))
    for i in range(s):
        line = random.sample(range(0, N), N)
        line.append(-1)
        pop[i] = line
    # print(pop)

    # 三、适应度函数计算
    for line in pop:
        sumdis = objective_func(line[:-1], dis)
        line[-1] = fitness_func(sumdis)
    # print('pop:', pop)

    # 四、选择算子
    # 本文采用最优保存策略
    print('=========选择=========')
    selec1, selec2, pop = selection(pop)

    # 五、 交配算子
    # 本文采用有序交叉法, 交配概率Pc(0.4, 0.99)
    print('=========交配=========')
    pop = crossover(selec1, selec2, N, s, pop)

    # 六、变异算子
    # 本文使用倒置变异法
    print('=========变异=========')
    pop = mutation(N, s, pop)


    # 七、终止条件，开始迭代
    answer = [] # 次优解序列
    for iter in range(iter_num):
        print("第{}次迭代：".format(iter))
        print('======适应度函数======')
        for line in pop:
            sumdis = objective_func(line[:-1], dis)
            line[-1] = fitness_func(sumdis)
        sumdis = objective_func(pop[selec1][:-1], dis)
        path = pop[selec1][:-1]
        answer.append(sumdis)
        print("此次次优解：{}".format(sumdis))
        print('=========选择=========')
        selec1, selec2, pop = selection(pop)
        print('=========交配=========')
        pop = crossover(selec1, selec2, N, s, pop)
        print('=========变异=========')
        pop = mutation(N, s, pop)

    print("answer:", answer)
    draw(X, Y, data_cities, answer, path)

if __name__=="__main__":
    main()


