# MNIST数据集分类_使用AdamOptIMizer法做优化
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
# one_hot 处理 -> 某个单位为1，其余全为零
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
# 总数据集  整除  批次大小
n_batch = mnist.train.num_examples // batch_size
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 输出层
# 权值
W = tf.Variable(tf.zeros([784, 10]))
# 偏置值
b = tf.Variable(tf.zeros([10]))
# 预测值
# 激活函数 -> softmax 交叉熵函数
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
# softmax 交叉熵代价函数
# 误差值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
'''
优化
'''
# 梯度下降法最小化误差值 loss
tran_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 使用AdamOptimizer方法最小化误差值 loss , 使用时学习率尽量小
# 1e-3 -> 10的-3次方
# tran_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
# 测试准确率 => 存放布尔值的列表
# tf.equal -> 比较参数一(真实值数据)和参数二(预测值数据)行[或者列]的最大值 => True or False
# argmax -> 返回一维张量中最大值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
# 求准确率
# tf.cast -> 将对比后的布尔值列表转换为对应的浮点值 => True为1.0，False为0
# tf.reduce_mean -> 平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 启动
with tf.Session() as sess:
    # 先初始化变量
    sess.run(tf.global_variables_initializer())
    # 周期
    for epoch in range(21):
        # 批次
        for batch in range(n_batch):
            # batch_xs
            # 获得一个批次，每次大小为100
            # 相当于每次获取100张图片
            # batch_ys
            # 图片的标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(tran_step, feed_dict={x:batch_xs, y:batch_ys})
        # 每阶段的测试准确率
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('第 %s 批次测试准确率为 %s' % (str(epoch), str(acc)))
