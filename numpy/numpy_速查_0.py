import numpy as np
import random
import time
from numpy import dot
from numpy import corrcoef, mean, std, multiply
from numpy import cov, sum, multiply, mean
from numpy import diag
from numpy import mat, linalg, transpose
from numpy import eye
from numpy import diag, linalg, mat
from numpy import mat
from numpy import mean
from numpy import multiply
from numpy import nonzero
from numpy import ones
from numpy import shape, array
from numpy import sum
from numpy import transpose, mat
from numpy import vdot
from numpy import zeros, ones
import tensorflow as tf
from numpy import arange

# xrange()是一个类，返回一个xrange()对象。Xrange()遍历后只返回一个值，
# range()遍历后返回一个列表，一次计算返回所有值。

x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[5,6], [7,8]], dtype=np.float64)
# print(x+y)
# print(np.add(x,y))

# print(x-y)
# print(np.subtract(x,y))

# print(x*y)
# print(np.multiply(x,y))

# print(x/y)
# print(np.divide(x,y))

# 开方
# print(np.sqrt(x))

# 矩阵乘
# [[1 * 5 + 2 * 7, 1 * 6 + 2 * 8], [3 * 5 + 4 * 7, 3 * 6 + 4 * 8]]
print(x.dot(y))

# std 标准方差
# var 方差

# 获取指定字符
for i in range(65, 91):
  # A - Z
  print(str(chr(i)))

# 相关系数矩阵
vc = [1,2,39,0,8]
vb = [1,2,38,0,8]
print(mean(multiply((vc-mean(vc)), (vb-mean(vb)))) / (std(vb) * std(vc)))
# corrcoef 得到相关系数矩阵(向量的相似程度)
print(corrcoef(vc, vb))

# 协方差
b = [1,3,5,6]
print(cov(b))
# 推
print(sum((multiply(b,b))-mean(b)*mean(b))/3)

# cov的参数是矩阵
# 输出结果也是矩阵
# 输出的矩阵为协方差矩阵
x = [[0, 1, 2],[2, 1, 0]]
print(cov(x))
print(sum((multiply(x[0],x[1]))-mean(x[0])*mean(x[1]))/2)

# 构建对角矩阵
d = [1, 2, 3]
dd = diag(d)
print(dd)

# 矩阵点积
# 矩阵乘
l = [[1,2,3], [4,5,6], [7,8,9]]
d = dot(l, 2)
print(d)
x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])
print(x.dot(y))

# 矩阵的特征值和特征向量
a = mat([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,4,0,0,0]])
b = a * a.T lamda

# 单元矩阵
print(eye(4))
print(eye(4, 3))

# 逆矩阵
d = [1,2,3]
dd = diag(d)
print(dd)
print(linalg.inv(dd))
print(mat(dd).I)

l = [[1,2,3], [4,5,6], [7,8,9]]
print(mat(l))


# 将列表转换成矩阵形式
a = [1,2,3]
b = [4,5,6]
s = [a, b]
print(mat(s))


# 均值
b = [1,3,5,6]
print(mean(b))
l = [[1,2,3,4,5,6],[3,4,5,6,7,8]]
# 全部元素求均值
print(mean(l))
# 按列求均值
print(mean(l, 0))
# 按行求均值
print(mean(l, 1))


# 乘
x1 = [1,2,3]
x2 = [4,5,6]

my = multiply(x1,x2)
print(my)


# 函数返回矩阵中非0元素的位置
x =[[1,0,0,0,2], [0,0,3,0,0]]
print(nonzero(x)[0])
'''
第一行是所有非零数所在行值
第二行是所有非零值所在列值
'''


# 维数
a = np.random.random((3, 4))
print(a)

# 取到2，从1开始取到3，含头不含尾
b = a[:2, 1:3]
print(b)

row_r1 = a[1, :]
row_r2 = a[1:2, :]
# 区别
# 使用切片语法访问数组时，得到的是原数组的子集
print(row_r1)
print(row_r2)
print('------------------------------')

ar = np.random.random((4, 2))
print(ar)
# 行(row) 列(col)
print(ar[[0,1,2], [0,1,0]])
print(ar[[0,0], [1,1]])
print('----------------------------------')

x = np.array([1,2])
# int32
print(x.dtype)
x = np.array([1.1,2.3])
# float64
print(x.dtype)
x = np.array([1,2], dtype=np.int64)
# int64
print(x.dtype)


# 指定行数全一矩阵
# 返回按要求的矩阵
ones = ones((2,1))
print(ones)


# 矩阵的行列数
a = array([[1,2,3], [4,5,6]])
print(a)
print(a.shape)


# 随机打乱列表中的元素数据
a = list(range(9))
print(a)
random.shuffle(a)
print(a)


# 将时间按照指定的格式转换
# import time
# import datetime, deate
# d = datetime.strptiime()
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


# 求和
x = [[0,1,2], [2,1,0]]
b = [1,3,5,6]
print(sum(b))
print(sum(x))
# 列
print(sum(x, 0))
# 行
print(sum(x, 1))


a = np.array([1, 2, 3])
print(type(a))
# 维数
print(a.shape)
a[0] = 4
print(a)

b = np.array([[1,2,3], [4,5,6]])
print(b)
# 矩阵维数
print(b.shape)
print(b[1][2], b[0][0])


# 将矩阵进行转置
# T
a = [[1], [2], [3]]
print(type(a))
a = mat(a)
print(type(a))
print(a)
print(a.transpose())
print(a.T)


# 元组
# 将数字转换为字符串
print(type(str(321)))


# 向量的点积
a = [1,2,3]
b = [4,5,6]
# 对应位置相乘求和
print(vdot(a, b))


# 指定行列全零矩阵
print(zeros((3, 2)))
# 全一
print(ones((3, 2)))
_x = tf.Variable(tf.zeros([2, 3]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  result = sess.run(_x)
  print(result)

# 可以互换指定区域的位置
l = [1,2,3,4,5,6]
print(l[3:6] + l[0:3])
# 成对获取x, y的值
a = [1,2,3]
b = [4,5,6]
for x, y in zip(a, b):
  print(x, y)


# 获取指定起始位置，指定步长的一系列数
d = 0.25
x = arange(-3.0, 3.0, d)
print(x)

# 测试range
for i in range(0, 5):
    print(i)


# 二进制和文本
# 二进制又分 numpy 专用的格式化二进制类型和无格式类型
# a = np.arange(0, 12)
a.shape = 3,4
print(a)
# tofile 将数组中的数据以二进制格式写入文件
a.tofile('a.bin')
# tofile 输出的数据没有格式，读取回来数据时需要格式化数据
b = np.fromfile('a.bin', dtype=np.int32)
print('b = %s' % b)
# 读入的时候设置正确的dtype和shape才能保证数据一致
# 按照原数据的shape修改读取后数据的shape
b.shape = 3, 4
print(b)
print('-----------------------')
# 使用load和save Numpy专用的二进制类型保存数据，会自动处理元素类型和shape的信息
np.save('a.npy', a)
c = np.load('a.npy')
print(c)
'''
将多个数组保存到一个文件中的话，可以使用numpy.savez函数。
savez函数的第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为arr_0, arr_1, ...。
savez函数输出的是一个压缩文件(扩展名为npz)，其中每个文件都是一个save函数保存的npy文件，文件名对应于数组名。
load函数自动识别npz文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容
'''
a = np.array([[1,2,3], [4,5,6]])
b = np.arange(0, 1.0, 0.1)
# 正弦
c = np.sin(b)
# 可以指定键 值， 也可以使用默认的 键
np.savez('result.npz', a, b_ = b, sin_array = c)
# 读取
r = np.load('result.npz')
print(r)
# a
print(r['arr_0'])
# b
print(r['b_'])
# c
print(r['sin_array'])


# 使用 savetxt 和 loadtxt 可以读写 1维 和 2维 的数组
# 广播 -> 4 行，-1(剩下的自动匹配)
a = np.arange(0, 12, 0.5).reshape(4, -1)
# 缺省(默认) 按'%.18e'格式保存数据，以空格分隔
np.savetxt('a.txt', a)
print(np.loadtxt('a.txt'))
# 修改为保存整数，以逗号分隔
np.savetxt('a.txt', a, fmt='%d', delimiter=',')
# 获取时需要使用 ，分隔
print(np.loadtxt('a.txt', delimiter=','))


# 将多个数组存储到一个npy文件中
a = np.arange(8)
print(a)
b = np.add.accumulate(a)
print(b)
c = a + b
f = file('result.npy', 'wb')
# 顺序保存a, b, c保存至文件对象 f
np.save(f, a)
np.save(f, b)
np.save(f, c)
f.close()
# 顺序读出
print(np.load(f))
print(np.load(f))
