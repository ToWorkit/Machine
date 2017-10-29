# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import leastsq
import matplotlib as mpl
import matplotlib.pyplot as plt
# 设置 FangSong/黑体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
# 解决 负号 的显示问题
mpl.rcParams['axes.unicode_minus'] = False

 http://old.sebug.net/paper/books/scipydoc/scipy_intro.html

'''
scipy中的子函数库optimize已经提供了实现最小二乘拟合算法的函数leastsq

要拟合的函数是一个正弦波函数，它有三个参数 A, k, theta ，分别对应振幅、频率、相角。
假设我们的实验数据是一组包含噪声的数据 x, y_1，其中y_1是在真实数据y_0的基础上加入噪声
通过leastsq函数对带噪声的实验数据x, y_1进行数据拟合，可以找到x和真实数据y_0之间的正弦关系的三个参数： A, k, theta
'''
# 最小二乘拟合
# S(\mathbf{p}) = \sum_{i=1}^m [y_i - f(x_i, \mathbf{p}) ]^2
def func(x, p):
    '''
    数据拟合所用的函数：A * sin(2 * pi * k * x + theta)
    '''
    A, k, theta = p
    return A * np.sin(2 * np.pi * k * x + theta)
def residuals(p, y, x):
    '''
    实验数据x, y和拟合函数之间的差，p 为拟合需要找到的系数
    '''
    return y - func(x, p)
x = np.linspace(0, -2 * np.pi, 100)
# 真实数据的函数参数
A, k, theta = 10, 0.34, np.pi / 6 
# 真实数据
y_0 = func(x, [A, k, theta])
# 加入噪声(干扰)之后的实验数据
# randn -> 从标准正态分布中返回一个或多个样本值
y_1 = y_0 + 2 * np.random.randn(len(x))

# 第一次猜测的函数拟合参数
p_0 = [7, 0.2, 0]

# 调用leastsq 进行数据拟合
# residuals -> 计算误差的函数
# p_0 -> 拟合参数的初始值
# args -> 需要拟合的实验数据
plsq = leastsq(residuals, p_0, args=(y_1, x))

print(u'真实参数:\t %s' % ([A, k, theta]))
# 实验数据拟合后的参数
print(u'拟合参数:\t %s' % plsq[0])
# print(plsq)

# 设置尺寸
plt.figure(figsize=(10,8))
plt.plot(x, y_0, label=u'真实数据')
plt.plot(x, y_1, label=u'带噪声(干扰)的实验数据')
plt.plot(x, func(x, plsq[0]), label=u'拟合数据')
# 图片的label放在右上角
plt.legend(loc='upper right')
plt.show()
# 拟合参数虽然和真实参数完全不同，但是由于正弦函数具有周期性，实际上拟合参数得到的函数和真实参数对应的函数是一致的



# http://old.sebug.net/paper/books/scipydoc/scipy_intro.html
# interpolate库提供了许多对数据进行 插值运算 的函数
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
# 设置 FangSong/黑体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
# 解决 负号 的显示问题
mpl.rcParams['axes.unicode_minus'] = False

# 使用直线和B-Spline对正弦波上的点进行插值
x = np.linspace(0, 2 * np.pi + np.pi / 4, 10)
y = np.sin(x)

x_new = np.linspace(0, 2 * np.pi + np.pi / 4, 100)
# interp1d -> 得到一个新的线性插值函数
f_linear = interpolate.interp1d(x, y)
# splrep -> B-Spline插值运算需要先使用splrep函数计算处B-Spline曲线的参数
tck = interpolate.splrep(x, y)
# 传递参数，计算处各个取样点的插值结果
y_bspline = interpolate.splev(x_new, tck)

plt.plot(x, y, 'o', label=u'原始数据')
plt.plot(x_new, f_linear(x_new), label=u'线性插值')
plt.plot(x_new, y_bspline, label=u'B-Spline插值')
plt.legend(loc='upper right')
plt.show()
