# encoding: utf-8
"""
@author: lin
@file: IcingThicknessPrediction.py
@time: 2018/10/11 16:43
@desc: Icing Thickness Prediction Model Using Fuzzy Logic Theory
"""
import numpy as np

"""基于模糊逻辑的覆冰预测模型
（1）四个输入：
环境温度（ET），环境湿度（EH），环境风速（EW），导线温度（CT）
论域范围：ET（-20～20℃），EH（0～100%），EW（0～20m/s）,CT（-20～20℃）
（2）模糊化：
隶属度函数--三角形函数
（3）模糊推理：
    1）语言变量：
        NB：负方向大（NegativeBig），NM：负方向中（NegativeMedium）,NS:负方向小（NegativeSmall）
        O:适中（Zero）
        PS:正方向小（PositiveSmall），PM：正方向中（PositiveMedium），PB：正方向大（PositiveBig）
    2）25条模糊规则
（4）去模糊化
（5）一个输出：
    覆冰厚度（IT）
    论域范围：IT（0-30mm）  
"""
# 三角函数类
class TriangleFunc:
    def __init__(self, low, high):
        self.area = (high-low) / 4
    def linear_func(self, x):
        return (1 / self.area) * x

    def traingle_func(self, x):
        x = x % self.area
        y = self.linear_func(x)
        return 1 - y, y

# 数据模糊化
def fuzzy(inputs):
    bounds = np.array([[-20, 20], [0, 100], [0, 20], [-20, 20]])
    # 数据标准化
    bounds[1] -= 50
    bounds[2] -= 10
    # print(bounds)
    inputs = inputs.T
    inputs[1] -= 50
    inputs[2] -= 10
    inputs = inputs.T
    # print(inputs)
    # 数据模糊化
    area = [] # 四个函数周期值
    tri_funcs = [] # 创建四个隶属度函数
    # 计算函数周期值和创建隶属度函数
    for i, bound in enumerate(bounds):
        area.append((bounds[i][1]-bounds[i][0]) / 4)
        tri_funcs.append(TriangleFunc(bounds[i][0], bounds[i][1]))
    # 创建隶属度矩阵，和相应的隶属度对应的模糊集矩阵
    member = []
    fuzzySet = []
    for input in inputs:
        y = []
        a = []
        for i, data in enumerate(input):
            y1, y2 = tri_funcs[i].traingle_func(data)# 可能出现y2==0
            a1 = data //area[i]
            a2 = a1 + 1 # 当a1 == 2, a2则越界,此时y2==0
            y.append((y1, y2))
            a.append((a1, a2))
        member.append(y)
        fuzzySet.append(a)
    # print(member)
    # print(fuzzySet)
    return np.array(member), np.array(fuzzySet)
# 激活规则，使用模糊规则
def active_rule(member, fuzzySet, rules):
    # 暂不考虑y2==0 和a2==3 的特殊情况
    newr = rules[:, : -1]
    outData = []
    for i in range(len(member)):
        data = []
        for idx0 in range(2):
            for idx1 in range(2):
                for idx2 in range(2):
                    for idx3 in range(2):
                        a = [member[i][0][idx0], member[i][1][idx1],
                             member[i][2][idx2], member[i][3][idx3]]
                        min = np.min(a)
                        f = np.array([fuzzySet[i][0][idx0], fuzzySet[i][1][idx1],
                             fuzzySet[i][2][idx2], fuzzySet[i][3][idx3]])
                        # print(f)
                        out = -10
                        for j, r in enumerate(newr):
                            if (f == r).all():
                                # print(r)
                                out = rules[j][-1]
                                break
                        # print(out)
                        data.append((min, out))
        outData.append(data)
    return np.array(outData)

# 去模糊化
def defuzzy(outData):
    w = [0, 6, 12, 18, 24]
    thickness = []
    print(outData)
    for data in outData:
        dic = {}
        max = []
        uper = 0
        for d in data:
            if d[1] == -10 : continue
            if (d[1] in dic)==False: dic[d[1]]= [d[0]]
            else: dic[d[1]].append(d[0])
        # print(dic)
        for key, value in dic.items():#key为模糊集，value为隶属度
            m = np.array(value).max()
            uper += m * w[int(key+2)]
            max.append(m)
        thickness.append(uper / np.sum(max))
    return thickness
def main():
    # (1)四个输入
    # ET,EH,EW,CT
    inputs = np.array([[-2, 79, 1, 1], [-3, 86,0.3,-5],[-2,97,6.8,1],[5,93,2,3],
                        [2,90,0.7,4],[-3,86,1.9,1],[-4,93,0.6,-1],[-3,97,0.3,-4],
                        [-4,96,0.5,-5],[-1,95,0.3,-2],[-2,84,0.8,-3],[1,80,1.5,-2],
                        [-6,94,1.7,-3],[-3,87,1,-3],[-1,96,0.3,-4],[0,93,0.5,-3]]
                     )
    real_out = np.array([5.89,4.42,13.46,3.35,6.86,4.38,8.87,\
                        7.12,9.39,7.1,5.66,5.46,9.33,3.89,6.74,4.48]
)
    rules = np.array(
            [[-1, 2, -1, -1, -1], [-1, 2, 0, -1, -2], [-1, 2, -1, 0, 0], [-1, 1, 0, 0, -1], [-1, 1, -2, 0, -1],
             [-1, 2, -2, -1, 1], [-1, -1, -1, 0, -2], [-1, 0, 0, -1, -1], [-1, 2, -2, -1, 2], [-1, 2, 0, 0, 1],
             [-1, -1, -1, 0, -2], [-2, 2, -2, -2, 2], [-1, 0, 1, 0, -2], [-1, 2, -2, -1, 0], [1, 2, -1, 0, -2],
             [1, 2, 0, 0, -2], [0, 2, -1, 0, -2], [-1, 2, -1, 0, -1], [0, 2, 0, 0, -1], [-1, 2, 0, 0, 0],
             [-1, 2, -1, -1, 1], [2, 2, -1, 2, -2], [2, 2, 0, 2, -2], [1, 1, 2, 1, -2], [0, -2, 0, 0, -2]]
                    )

    m ,f =fuzzy(inputs)
    outData = active_rule(m, f, rules)
    pre_thickness = defuzzy(outData)
    print("real,pre,abs_error:")
    for r, p in zip(real_out,pre_thickness):
        print(r, ' ', p, ' ', abs(r-p))

if __name__ == "__main__":
    main()