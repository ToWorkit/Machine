import numpy as np
'''
SciPy函数库在NumPy库的基础上增加了众多的数学、科学以及工程计算中常用的库函数。
例如线性代数、常微分方程数值求解、信号处理、图像处理、稀疏矩阵等等。
使用SciPy进行插值处理、信号滤波以及用C语言加速计算
http://old.sebug.net/paper/books/scipydoc/scipy_intro.html
'''
# 最小二乘拟合
# S(\mathbf{p}) = \sum_{i=1}^m [y_i - f(x_i, \mathbf{p}) ]^2

# randn -> 从标准正态分布中返回一个或多个样本值
print(np.random.randn(10))
# 两行四列
print(np.random.randn(2, 4))

# 通过求解卷积的逆运算演示fmin的功能
import scipy.optimize as opt
import numpy as np
# optimize库提供了几个求函数最小值的算法：fmin, fmin_powell, fmin_cg, fmin_bfgs
# http://old.sebug.net/paper/books/scipydoc/scipy_intro.html
def test_fmin_convolve(fminfunc, x, h, y, y_n, x_0):
    '''
    x (*) h = y, (*) -> 卷积
    y_n -> 在y的基础上添加一些干扰噪声的结果
    x_0 -> 求解x的初始值
    '''
    def convolve_func(h):
        '''
        计算 y_n - x (*) h 的 power(功率)
        fmin 将通过计算使得此power最小
        '''
        return np.sum((y_n - np.convolve(x, h)) ** 2)
    # 调用fmin函数，以 x_0 为初始值
    h_0 = fminfunc(convolve_func, x_0)
    
    print(fminfunc.__name__)
    print('----------------')
    # 输出 x (*) h_0 和 y 之间 的相对误差
    print('x (*) h_0 和 y 之间 的相对误差(error of y):', np.sum((np.convolve(x, h_0) - y) ** 2) / np.sum(y ** 2))
    # 输出 h_0 和 h 之间的相对误差
    print('h_0 和 h 之间的相对误差(error of h):', np.sum((h_0 -h) ** 2) / np.sum(h ** 2))
    print('=================')
def test_n(m, n, nscale):
    '''
    随机产生x, h, y, y_n, x_0等数列，调用各种fmin函数求解b
    m 为 x 的长度，n 为 h 的长度， nscale为干扰的强度
    '''
    x = np.random.rand(m)
    h = np.random.rand(n)
    y = np.convolve(x, h)
    y_n = y + np.random.rand(len(y)) * nscale
    x_0 = np.random.rand(n)
    
    # optimize库提供了几个求函数最小值的算法
    test_fmin_convolve(opt.fmin, x, h, y, y_n, x_0)
    test_fmin_convolve(opt.fmin_powell, x, h, y, y_n, x_0)
    test_fmin_convolve(opt.fmin_cg, x, h, y, y_n, x_0)
    test_fmin_convolve(opt.fmin_bfgs, x, h, y, y_n, x_0)

if __name__ == '__main__':
    test_n(200, 20, 0.1)


# http://old.sebug.net/paper/books/scipydoc/scipy_intro.html
# optimize库中的fsolve函数可以用来对非线性方程组进行求解
# fsolve(func, x0)
'''
func(x)是计算方程组误差的函数，它的参数x是一个矢量，表示方程组的各个未知数的一组可能解，func返回将x代入方程组之后得到的误差；x0为未知数矢量的初始值。如果要对如下方程组进行求解的话：

f1(u1,u2,u3) = 0
f2(u1,u2,u3) = 0
f3(u1,u2,u3) = 0
那么func可以如下定义：

def func(x):
    u1,u2,u3 = x
    return [f1(u1,u2,u3), f2(u1,u2,u3), f3(u1,u2,u3)]
'''
# 求解如下方程组的解
'''
5 * x_1 + 3 = 0
4 * x_0 * x_0 - 2 * sin(x_1 * x_2) = 0
x_1 * x_2 - 1.5 = 0
'''
from scipy.optimize import fsolve
from math import sin, cos
def foo(x):
    x_0 = float(x[0])
    x_1 = float(x[1])
    x_2 = float(x[2])
    return [
        5 * x_1 + 3,
        4 * x_0 * x_0 - 2 * sin(x_1 * x_2),
        x_1 * x_2 - 1.5
    ]
result = fsolve(foo, [1, 1, 1])
print(result)
print('')
# 雅可比矩阵 -> http://old.sebug.net/paper/books/scipydoc/scipy_intro.html
print(foo(result))
