# 梯度下降法
import tensorflow as tf
import numpy as np
'''
样本
'''
# 随机生成100个点
x_data = np.random.rand(100)
# 平面坐标系中接近于一条直线, 斜率 0.1 截距 0.2
# 样本真实值
y_data = x_data * 0.1 + 0.2
'''
模型
'''
# 构造一个线性模型并初始化值(可根据需求设置初始值)
# 斜率
k = tf.Variable(1.3)
# 截距
b = tf.Variable(4.6)
# 模型预测值
y = k * x_data + b
'''
优化
训练并优化模型使之无限接近(=>)样本数据(k => 样本斜率, b => 样本截距)
'''
# 二次代价函数
# reduce_mean -> 平均值
# square -> 平方
# y_data -> 真实值
# y -> 预测值
# y_data - y -> 误差值
loss = tf.reduce_mean(tf.square(y_data - y))

# 梯度下降法 -> 用作训练的优化器
# arguments 学习率
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
# 训练 -> 目的最小化loss(误差值)
# 误差值越小预测值越接近真实值
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 创建会话
with tf.Session() as sess:
    # 执行初始化
    sess.run(init)
    # 迭代 -> 训练次数
    for step in range(201):
        # 执行训练
        sess.run(train)
        # 每二十次打印一次
        if  step % 20 == 0:
            print(step, sess.run([k, b]))


'''

'''

# 非线性回归_有神经网络初级相关的详细注释
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
样本
'''
# 随机生成200个随机点
# -0.5 - 0.5 的范围内 均匀分布
# 转换维度为 2维
# newaxis -> 插入维度
# 200 行 1 列
x_data = np.linspace(-0.5,0.5,200)[:, np.newaxis]
# 干扰项
# shape -> 矩阵的维度
# 形状与 x_data 一致
# np.random.normal -> 正态分布（随机抽样）
noise = np.random.normal(0,0.02,x_data.shape)
# 真实值
# x_data 实际为 U 型图， 加入干扰后散点会上下浮动
# x_data的平方加上随机干扰项
y_data = np.square(x_data) + noise
'''
输入层
'''
# 输入
# 定义两个placeholder(占位符) -> 依据样本定义
# 类型为32位浮点型，形状 -> 行 不确定，列 1列 
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
'''
中间层
'''
# 定义神经网络中间层
# 都初始化为 0 tf.zeros，并不是很好
# 权值 -> 连接输入层和中间层
# 1 个输入层神经元，10 个中间层神经元
# tf.random_normal -> 从正态分布中输出随机值
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
# 偏置值
# 全零矩阵，1 行 10 列
biases_L1 = tf.Variable(tf.zeros([1,10]))
# 信号总和
# matmul -> 矩阵相乘
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
# 中间层输出
# 激活函数 -> 用双曲正切函数来作为激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)
'''
输出层
'''
# 定义神经网络输出层
# 权值
# 10 个中间层神经元，1 个输出层神经元
Weights_L2 = tf.Variable(tf.random_normal([10,1])) 
# 偏置值
biases_L2 = tf.Variable(tf.zeros([1,1]))
# 信号总和
# matmul -> 矩阵相乘
# 输出层的输入就相当于是中间层的输出 -> L1
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 激活函数 -> 选用双曲正切函数
# 预测结果
prediction = tf.nn.tanh(Wx_plus_b_L2)
'''
优化
'''
# 二次代价函数
# 误差值 = 真实值 - 预测值 的 平方 的 平均值
loss = tf.reduce_mean(tf.square(y - prediction))
# 优化 -> 梯度下降法
# 0.1 的学习率最小化误差值
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练次数
    for _ in range(2000):
        # 传入样本值
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    # 画图
    plt.figure()
    # 样本
    plt.scatter(x_data, y_data)
    # 预测结果
    # 样本值， 预测值， 线的颜色实线， 线宽 5
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

'''

'''

# MNIST数据集分类_手写识别_初版_无隐藏层_无优化
import tensorflow as tf
# 手写数字相关工具包
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
# 没有的话会自动去下载数据集
# 数据路径  转为one_hot(某一位数字为1，其余数字都为0)格式
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 每个批次的大小
'''
优化
    可修改批次大小
    可添加隐藏层
'''
# 每次放入批次大小的数据集
# 形式为矩阵
batch_size = 100
# 批次的个数
# 计算一共有多少个批次
# 总训练集  整除  批次大小
n_batch = mnist.train.num_examples // batch_size
# 输入
# 定义两个placeholder
# [行 -> 任意值(此处为100，与上面的批次大小一致), 列 -> 每张图片都是28*28，需要转为一维的向量也就是28*28=784]
x = tf.placeholder(tf.float32, [None, 784])
# 数字为 0-9 
y = tf.placeholder(tf.float32, [None, 10])
# 输出
# 创建简单的神经网络
'''
可修改初始值 ?
'''
# 权值 -> 当前权值 784个输入层， 10个输出层
W = tf.Variable(tf.zeros([784,10]))
# 偏置值
b = tf.Variable(tf.zeros([10]))
# 预测值
# 激活函数 -> 需用softmax函数
# 数据和权值矩阵相乘 + 偏置值 再 使用softmax函数激活
# softmax -> 转换为概率值
prediction = tf.nn.softmax(tf.matmul(x,W) + b)
# # 输出
# # 创建简单的神经网络
# '''
# 可修改初始值 ?
# '''
# # 权值 -> 当前权值 784个输入层， 10个输出层
# W = tf.Variable(tf.zeros([784,10]))
# # 偏置值
# # 真实值
# b = tf.Variable(tf.zeros([10]))
# # 预测值
# # 激活函数 -> 需用softmax函数
# # 数据和权值矩阵相乘 + 偏置值 再 使用softmax函数激活
# # softmax -> 转换为概率值
# prediction = tf.nn.softmax(tf.matmul(x,W) + b)
# 优化
'''
可选择优化方式，比如交叉熵
'''
# 二次代价函数
# 误差值
# 真实值 - 预测值 的 平方 的 平均值
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法优化 loss
''' 
学习率可修改
'''
# 0.2的学习率最小化loss
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 测试准确率 => 存放布尔值的列表
# tf.equal -> 比较参数一(真实值数据)和参数二(预测值数据)行[或者列]的最大值 => True or False
# argmax -> 返回一维张量中最大值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
# 求准确率
# tf.cast -> 将对比后的布尔值列表转换为对应的浮点值 => True为1.0，False为0
# tf.reduce_mean -> 平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    # 先初始化变量
    sess.run(tf.global_variables_initializer())
    # 21个周期
    # 图片训练 21 个周期
    for epoch in range(21):
        # 批次
        # 所有的图片都训练一次
        for batch in range(n_batch):
            # batch_xs
            # 获得一个批次，每次大小为100
            # 相当于每次获取100张图片
            # batch_ys
            # 图片的标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 执行训练传入数据
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys})
        # 每个周期的准确率
        # 测试集的图片和测试集的图片标签
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('(Iter)第\t%s\t个周期(Testing Accuracy)准确率\t%s' % (str(epoch), str(acc)))

'''

'''
