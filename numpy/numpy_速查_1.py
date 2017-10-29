import numpy as np
import math 
import time

# range() 创建int类类 => list
# arange() 可以使用float类型 => array
# 0 - 60 (含头不含尾) 步长为 10  reshape -> 给予数组一个新的形状，而不改变它的数据
a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
print(a)
# 下面俩和上面对比着看就明白了
print(np.arange(6))
print(np.arange(0, 60, 10))

# 需要3个指针和3个整数对象，对于数值运算比较浪费内存和cpu
L = [1, 2, 3]
# 所以使用Numpy提供的ndarray对象 -> 存储单一数据类型的多维数组
_L = np.array(L)
print(_L)
# 获取长度
print(_L.shape)
b = np.array([[1,2], [2,3], [4,5], [6,7], [8,9], [10, 11]])
print(b)
print(b.shape)
# 可以强制性修改shape, 当某个轴为-1时，将根据数组元素的个数自动计算此轴的长度
b.shape = 2, -1
print(b)
print(b.shape)
# 使用reshape方法，可以创建改变了尺寸的新数组，原数组的shape保持不变, 并且reshape之后的数组与原数组更享内存数据(指向一致)
c = b.reshape((3, -1))
print(c)
# 数据类型
print(c.dtype)
# 指定数据类型
d = np.array([[2,3], [4,5]], dtype=np.float)
print(d)
# astype -> 数据类型转换
nf = d.astype(np.int)
print(nf)

# 线性
# linspace函数创建数组, params -> 初值， 终值，元素个数， 是否包含终值(缺省为True)
a = np.linspace(1, 10, 10)
print(a)
# 不包含终值
b = np.linspace(1, 10, 10, endpoint=False)
print(b)


# 对数
# 创建等比数列
# 10 ** 2 10的2次方
# 现实计算中 10^2 为10的2次方，但实际在算法中不是，上面的才是
# 当前起始值为 10**1， 终止值为 10**2, 有9个数的等比数列, 默认包含终止值
d = np.logspace(1, 2, 9)
# 10**1 - 10**4 
_d = np.logspace(1, 4, 4)
print(d)
print(_d)
print(10**3)
print(10^3)
# 创建基础值， 默认为10
# 2 ** 0  - 2 ** 10
g = np.logspace(0, 10, 11, base=2)
print(g)
print(2**0)
print(2**10)

# 使用frombuffer, fromstring, fromfile等函数可以从字节序列创建数组
st = 'abcd'
# 对应ASCALL码值
g = np.fromstring(st, dtype=np.int8)
print(g)


# 存取
# 含头不含尾
a = np.arange(10)
print(a)
print(a[3])
# 切片[3:5] -> 含头不含尾，下标从第3位开始截取到第5位
print(a[3:5])
# 省略开始，则从0开始, 或者省略最后
print(a[:3])
print(a[3:])
# 步长为2(相隔2位)
print(a[2:8:2])
# 头尾省略，步长为-1 -> 翻转数组
print(a[::-1])
# 切片的数据和原始数据指向一致，共享数据
a[1:4] = 10, 20, 30
print(a)

b = a[1:4]
b[0] = 200
print(a)
print(b)


# 根据整数数组存取
# 当使用整数序列对数组元素进行存取时，将整个整数序列中的每个元素作为下标，整数序列可以是列表或者是数组
# 使用整数序列作为下标获得的数组不和原始数组共享空间
a = np.logspace(0, 9, 10, base=2)
print(a)
i = np.arange(0, 10, 2)
print(i)
# 利用i做下标取a的数据
b = a[i]
print(b)
b[3] = 333
print(b)
print(a)


# 随机生成 10 个 0 - 1 中均匀分布的随机数 
a = np.random.rand(10)
print(a)
print(a > 0.5)
# 取出大于0.5的项
b = a[a > 0.5]
print('b = %s' % b)
# 将原始数据中 大于 0.5 的数据都强制指定为 0.5
a[a > 0.5] = 0.5
# b 不会受影响
print(a)
print(b)


# 行向量
a = np.arange(0, 60, 10)
print(a)
# 转为列向量
b = a.reshape((-1,1))
print(b)
# 行加列
f = a + b
print(f)
# 合并上面的代码
a = np.arange(0, 60, 10).reshape((-1, 1))+np.arange(6)
print(a)
# 二维数组切片 行 列
# 第 0 行的第 2 个下标
print(a[[0,1,2], [2,3,4]])
# 行步长， 列步长
print(a[::2, ::2])


# 定义结构数组
personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    # formats -> 格式
    # str32  
    # int32
    # float32
    'formats': ['S32', 'i', 'f']
})
a = np.array([('LaoPo', 27, 60.9), ('Me', 24, 64.1)], dtype=personType)
# b -> byte 字节字符
print(a)
print(a[1])
# 获取后数据共享
m = a[1]
print(m)
m['name'] = 'My'
print(a)
print(m)


a = np.array([[0,1,2], [3,4,5], [6,7,8]], dtype=np.float32)
print(a)
# 取
# 行步长， 列步长
b = a[::2, ::2]
print(b)


# linspace函数创建数组, params -> 初值， 终值，元素个数， 是否包含终值(缺省为True)
x = np.linspace(0, 5, 5)
print(x)
# ufunc函数 -> universal function
# 基于c实现，速度快
# 对数组中的每个元素进行正弦计算
y = np.sin(x)
print(y)


# 对比numpy.math和python标准库的math.sin计算速度
x = [i * 0.001 for i in range(1000000)]
start = time.clock()
# enumerate -> 枚举
# 遍历
for i, t in enumerate(x):
    x[i] = math.sin(t)
print('math.sin => %s' % (float(time.clock()) - float(start)))

x = np.array(x)
start = time.clock()
np.sin(x)
print('np.sin => %s' % (float(time.clock()) - float(start)))


a = np.arange(0, 4)
print(a)
b = np.arange(3, 7)
print(b)
# 对应元素的和
c = np.add(a, b)
print(c)

# axis
# 列
a = a.repeat(5, axis=1)
print(a)
# 行
b = b.repeat(6, axis=0)
print(b)
# 行
print(np.add.reduce([[1,2,3], [4,5,6]]))
# 列
print(np.add.reduce([[1,2,3], [4,5,6]], axis=1))


# ogrid -> 像一个多维数组一样，用切片组元作为下标进行存取，返回的是一组可以用来广播计算的数组
# 开始值:结束值:步长，和np.arange(开始值, 结束值, 步长)类似
x, y = np.ogrid[0:5, 0:5]
print(x)
print(y)
# 开始值:结束值:长度j，当第三个参数为虚数时，它表示返回的数组的长度，和np.linspace(开始值, 结束值, 长度)类似
a, b = np.ogrid[0:1:4j, 0:1:3j]
print(a)
print(b)


a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(5)
print(a)
# 二维数组切片
# 第 0 行 第 2 位下标
print(a[[0, 1, 2], [2, 3, 4]])
# 第 4 行 的 2，3, 4下标
print(a[4, [2, 3, 4]])
# 从第 4 行开始后面的都要
print(a[4:, [2, 3, 4]])

# 布尔值索引
i = np.array([True, True, False, True, False, False])
print(a[i])
# 取所有行的第 3 个下标
print(a[i, 3])


# 元素去重
a = np.array((1, 1, 2, 1, 1, 3, 4, 5, 6, 4, 3, 2))
print(np.unique(a).reshape(3,-1))
# 二维数组去重
b = np.array(((1, 2), (2, 3), (1, 2), (3,4)))
print(b)
# 先取出放入元组中，然后就可以在放入set集合中，然后利用集合的特性去重
print('去重: %s \n' % np.array(list(set([tuple(t) for t in b]))))


# 矩阵相乘
a = np.arange(1, 10) .reshape(3, 3)
print(a)
b = a + 10
print(b)
print('-----------------------')
# a 的第一行 乘以 b 的第一列 再求和
print(np.dot(a, b))
# 对应位置相乘
print('------------------------')
print(a * b)
print(np.multiply(a, b))


# 拼接
a = np.arange(1, 10)
b = np.arange(20, 30)
print(np.concatenate((a, b)))


'''
矩阵运算
'''
# 划重点 -> 不推荐在较复杂的程序中使用 matrix
a = np.matrix([[1,2,3], [5,5,6], [7,9,9]])
# a ** -1  -> a 的 逆矩阵
print(a * a ** -1)
# reshape -> 转为二维 -> 3行5列
b = np.arange(15).reshape(3,5)
print(b)
# T 转置
# 5行3列
print(b.T)

# 矩阵的乘积可以使用dot函数进行计算。
# 对于二维数组，它计算的是矩阵乘积，对于一维数组，它计算的是其点积。
# 将一维数组当作列矢量或者行矢量进行矩阵运算时， 使用reshape函数将一维数组转换为二维数组
'''
dot : 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)；
      对于二维数组，计算的是两个数组的矩阵乘积；
      对于多维数组，它的通用计算公式如下，即结果数组中的每个元素都是：数组a的最后一维上的所有元素与数组b的倒数第二位上的所有元素的乘积和
      dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
'''
a = np.arange(12).reshape(2,3,2)
print(a)
print('---------------')
b = np.arange(12, 24).reshape(2,2,3)
print(b)
print('---------------')
c = np.dot(a, b)
print(c)


# inner : 和dot乘积一样，对于两个一维数组，计算的是这两个数组对应下标元素的乘积和；
# 对于多维数组，它计算的结果数组中的每个元素都是：
# 数组a和b的最后一维的内积，因此数组a和b的最后一维的长度必须相同
# inner(a, b)[i,j,k,m] = sum(a[i,j,:]*b[k,m,:])

# outer : 只按照一维数组进行计算，如果传入参数是多维数组，则先将此数组展平为一维数组之后再进行运算。
# outer乘积计算的列向量和行向量的矩阵乘积
print(np.outer([1,2,3], [4,5,6,7]))

'''
矩阵中更高级的一些运算可以在NumPy的线性代数子库linalg中
inv函数计算逆矩阵，solve函数可以求解多元一次方程组
solve函数有两个参数a和b。a是一个N*N的二维数组，而b是一个长度为N的一维数组，solve函数找到一个长度为N的一维数组x，使得a和x的矩阵乘积正好等于b，数组x就是多元一次方程组的解
'''
# 10行10列
a = np.random.rand(10, 10)
print(a)
