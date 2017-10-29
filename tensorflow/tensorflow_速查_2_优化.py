# MNIST数据集分类_手写识别_交叉熵版
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
# [行 -> 任意值(与传入的批次大小一致), 列 -> 每张图片都是28*28，需要转为一维的向量也就是28*28=784]
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
# 真实值
b = tf.Variable(tf.zeros([10]))
# 预测值
# 激活函数 -> 需用softmax函数
# 数据和权值矩阵相乘 + 偏置值 再 使用softmax函数激活
# softmax -> 转换为概率值
prediction = tf.nn.softmax(tf.matmul(x,W) + b)
# 优化
'''
可选择优化方式，比如交叉熵
'''
# 二次代价函数
# 误差值
# 真实值 - 预测值 的 平方 的 平均值
# loss = tf.reduce_mean(tf.square(y - prediction))

# softmax交叉熵代价函数
# 标签值(真实值), 预测值  再求 平均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降法优化
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

# MNIST数据集分类_手写识别_交叉熵_使用Dropout防止过拟合
# 防止过拟合
# Dropout -> 迭代过程中使得部分神经元工作，部分神经元不工作，每迭代一次就更换一次
import tensorflow as tf
# 手写数字相关工具包
from tensorflow.examples.tutorials.mnist import input_data
# 样本
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
# 输入层
# 定义两个placeholder
# [行 -> 任意值(此处为100，与上面的批次大小一致), 列 -> 每张图片都是28*28，需要转为一维的向量也就是28*28=784]
x = tf.placeholder(tf.float32, [None, 784])
# 数字为 0-9 
y = tf.placeholder(tf.float32, [None, 10])
'''
增加一个输入，用来设置Dropout的参数
'''
keep_prob = tf.placeholder(tf.float32)
# 创建简单的神经网络

# 隐藏层(中间层)
# # 都初始化为 0 tf.zeros，并不是很好
# '''
# 可修改初始值 ?
# '''
# # 权值 -> 当前权值 784个输入层， 10个输出层
# W = tf.Variable(tf.zeros([784,10]))
# # 偏置值
# # 真实值
# b = tf.Variable(tf.zeros([10]))

'''
初始化权值
'''
# 截断的正态分布， param => (限制值(784 - 2000个神经元)， 标准差为0.1)
W1 = tf.Variable(tf.truncated_normal([784, 2000],stddev=0.1))
# 偏置值
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
# 隐藏层输出
# 使用双曲正切函数激活
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 防止过拟合 -> Dropout
# param => (某一层的输出， 设置参数(%) -> 代表有多少神经元是工作的 1 => 100% 0.5 => 50%)
L1_drop = tf.nn.dropout(L1, keep_prob)

'''
继续添加隐藏层的神经网络元是为了测试过拟合
'''
# 截断的正态分布， param => (限制值， 标准差为0.1)
W2 = tf.Variable(tf.truncated_normal([2000, 2000],stddev=0.1))
# 偏置值
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
# 隐藏层输出
# 使用双曲正切函数激活
# tf.matmul(L1_drop, W2) -> 信号总和 连接 上一个隐藏层
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# 防止过拟合 -> Dropout
# param => (某一层的输出， 设置参数(%) -> 代表有多少神经元是工作的 1 => 100% 0.5 => 50%)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 截断的正态分布， param => (限制值， 标准差为0.1)
W3 = tf.Variable(tf.truncated_normal([2000, 1000],stddev=0.1))
# 偏置值
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
# 隐藏层输出
# 使用双曲正切函数激活
# tf.matmul(L2_drop, W3) -> 信号总和 连接 上一个隐藏层
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# 防止过拟合 -> Dropout
# param => (某一层的输出， 设置参数(%) -> 代表有多少神经元是工作的 1 => 100% 0.5 => 50%)
L3_drop = tf.nn.dropout(L3, keep_prob)
#输出层
'''
初始化权值
'''
# 截断的正态分布， param => (限制值， 标准差为0.1)
# 从截断的正态分布中输出随机值
W4 = tf.Variable(tf.truncated_normal([1000, 10],stddev=0.1))
# 偏置值
b4 = tf.Variable(tf.zeros([10]) + 0.1)
# 预测值
# 激活函数 -> 需用softmax函数
# 数据和权值矩阵相乘 + 偏置值 再 使用softmax函数激活
# softmax -> 转换为概率值
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4) + b4)
# 优化
'''
可选择优化方式，比如交叉熵
'''
# 二次代价函数
# 误差值
# 真实值 - 预测值 的 平方 的 平均值
# loss = tf.reduce_mean(tf.square(y - prediction))

# softmax交叉熵
# 标签值(真实值), 预测值 传入之后再求平均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降法优化
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
    for epoch in range(31):
        # 批次
        # 所有的图片都训练一次
        for batch in range(n_batch):
            # batch_xs
            # 获得一个批次，每次大小为100
            # 相当于每次获取100张图片
            # batch_ys
            # 图片的标签
            # 训练时使用训练集数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 执行训练传入数据
#             sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys,keep_prob:1.0})
            # 启用Dropout传入0.7 -> 70%的神经元参与训练
            # 结果分析使用Dropout可以使测试准确率和训练准确率的差距缩小
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys,keep_prob:0.7})
        # 两类数据，一类测试集， 一类训练集    
        # 测试模型的好坏使用测试集数据
        # 每个周期的准确率
        # 测试集的图片和测试集的图片标签
#         test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1.0})
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:0.7})
        # 训练集
        # 训练时使用到的数据
#         train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:0.7})
        print('(Iter)第\t%s\t个周期(Testing Accuracy)测试准确率\t%s\t(Testing Accuracy)训练准确率\t%s\t' % (str(epoch), str(test_acc), str(train_acc)))

'''

'''

# MNIST数据集分类_手写识别_交叉熵_使用Dropout防止过拟合_继续优化
# 防止过拟合
# Dropout -> 迭代过程中使得部分神经元工作，部分神经元不工作，每迭代一次就更换一次
import tensorflow as tf
# 手写数字相关工具包
from tensorflow.examples.tutorials.mnist import input_data
# 样本
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
# 输入层
# 定义两个placeholder
# [行 -> 任意值(此处为100，与上面的批次大小一致), 列 -> 每张图片都是28*28，需要转为一维的向量也就是28*28=784]
x = tf.placeholder(tf.float32, [None, 784])
# 数字为 0-9 
y = tf.placeholder(tf.float32, [None, 10])
'''
增加一个输入，用来设置Dropout的参数
'''
keep_prob = tf.placeholder(tf.float32)
'''
增加一个学习率的变量
'''
lr = tf.Variable(0.001, dtype=tf.float32)
# 创建简单的神经网络

# 隐藏层(中间层)
# # 都初始化为 0 tf.zeros，并不是很好
# '''
# 可修改初始值 ?
# '''
# # 权值 -> 当前权值 784个输入层， 10个输出层
# W = tf.Variable(tf.zeros([784,10]))
# # 偏置值
# # 真实值
# b = tf.Variable(tf.zeros([10]))

'''
初始化权值
'''
# 截断的正态分布， param => (限制值(784 - 2000个神经元)， 标准差为0.1)
W1 = tf.Variable(tf.truncated_normal([784, 1000],stddev=0.1))
# 偏置值
b1 = tf.Variable(tf.zeros([1000]) + 0.1)
# 隐藏层输出
# 使用双曲正切函数激活
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 防止过拟合 -> Dropout
# param => (某一层的输出， 设置参数(%) -> 代表有多少神经元是工作的 1 => 100% 0.5 => 50%)
L1_drop = tf.nn.dropout(L1, keep_prob)

'''
继续添加隐藏层的神经网络元是为了测试过拟合
'''
# 截断的正态分布， param => (限制值， 标准差为0.1)
W2 = tf.Variable(tf.truncated_normal([1000, 500],stddev=0.1))
# 偏置值
b2 = tf.Variable(tf.zeros([500]) + 0.1)
# 隐藏层输出
# 使用双曲正切函数激活
# tf.matmul(L1_drop, W2) -> 信号总和 连接 上一个隐藏层
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# 防止过拟合 -> Dropout
# param => (某一层的输出， 设置参数(%) -> 代表有多少神经元是工作的 1 => 100% 0.5 => 50%)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 截断的正态分布， param => (限制值， 标准差为0.1)
W3 = tf.Variable(tf.truncated_normal([500, 300],stddev=0.1))
# 偏置值
b3 = tf.Variable(tf.zeros([300]) + 0.1)
# 隐藏层输出
# 使用双曲正切函数激活
# tf.matmul(L2_drop, W3) -> 信号总和 连接 上一个隐藏层
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# 防止过拟合 -> Dropout
# param => (某一层的输出， 设置参数(%) -> 代表有多少神经元是工作的 1 => 100% 0.5 => 50%)
L3_drop = tf.nn.dropout(L3, keep_prob)
#输出层
'''
初始化权值
'''
# 截断的正态分布， param => (限制值， 标准差为0.1)
# 从截断的正态分布中输出随机值
W4 = tf.Variable(tf.truncated_normal([300, 10],stddev=0.1))
# 偏置值
b4 = tf.Variable(tf.zeros([10]) + 0.1)
# 预测值
# 激活函数 -> 需用softmax函数
# 数据和权值矩阵相乘 + 偏置值 再 使用softmax函数激活
# softmax -> 转换为概率值
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4) + b4)
# 优化
'''
可选择优化方式，比如交叉熵
'''
# 二次代价函数
# 误差值
# 真实值 - 预测值 的 平方 的 平均值
# loss = tf.reduce_mean(tf.square(y - prediction))

# softmax交叉熵代价函数
# 标签值(真实值), 预测值 传入之后再求平均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降法训练
''' 
学习率可修改
'''
# 0.2的学习率最小化loss
'''
使用时传入lr学习率， 优化loss，学习率最低的地方就是loss最低的地方也就是准确率最高的地方
'''
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
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
    for epoch in range(41):
        '''
        传入学习率
        训练之初模型比较混乱，所以给一个比较大的学习率以达到快速收敛的目的，然后再逐渐的降低达到最低的点
        '''
        # ref value => 将 value(** -> 次方) 赋值 ref
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        # 批次
        # 所有的图片都训练一次
        for batch in range(n_batch):
            # batch_xs
            # 获得一个批次，每次大小为100
            # 相当于每次获取100张图片
            # batch_ys
            # 图片的标签
            # 训练时使用训练集数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 执行训练传入数据
#             sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys,keep_prob:1.0})
            # 启用Dropout传入0.7 -> 70%的神经元参与训练
            # 结果分析使用Dropout可以使测试准确率和训练准确率的差距缩小
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys,keep_prob:0.7})
        '''
        学习率
        '''
        learning_rate = sess.run(lr)
        # 两类数据，一类测试集， 一类训练集    
        # 测试模型的好坏使用测试集数据
        # 每个周期的准确率
        # 测试集的图片和测试集的图片标签
#         test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1.0})
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:0.7})
        # 训练集
        # 训练时使用到的数据
#         train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:0.7})
        print('(Iter)第 %s 个周期(Testing Accuracy)测试准确率 %s (Testing Accuracy)训练准确率 %s (Learning Rate)学习率 %s ' % (str(epoch), str(test_acc), str(train_acc), str(learning_rate)))
