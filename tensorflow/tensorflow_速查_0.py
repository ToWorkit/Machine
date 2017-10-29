import tensorflow as tf
a = tf.constant(10)
b = tf.constant(22)

with tf.Session() as sess:
    # 从正态分布中输出随机值
    _random_normal = sess.run(tf.random_normal([1,10]))
    print(_random_normal)
    # 从截断的正态分布中输出随机值
    _truncated_normal = sess.run(tf.truncated_normal([1,10]))
    print(_truncated_normal)

import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [4, 7, 1], [6, 7, 8]])
# dot 求阶乘
d = np.dot(a, b)
print(d)

l = np.array([[1, 2, 3], [4, 5, 6]])
t = np.array([[7, 8, 9], [4, 7, 1]])
# multiply 求对应位置的乘积
c = np.multiply(l, t)
print(c)


import tensorflow as tf
# 定义变量
_v = tf.Variable([1,2])
# 定义常量
_c = tf.constant([3,4])
# 减法操作
sub = tf.subtract(_v, _c)
# 加法操作
add = tf.add(_c, sub)
# 必须操作
# 初始化变量
init = tf.global_variables_initializer()
# 建立会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

# 变量还有一个name的参数
state = tf.Variable(0,name='counter')
# 加 1 op
add_value = tf.add(state,1)
# tf 的赋值操作，不是 =, 将add_value的值赋给state
update = tf.assign(state,add_value)
# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 自增之前的
    print('自增之前：%s' % sess.run(state))
    # 来五次，我能打十个
    for i in range(0,5):
        print(sess.run(update))


import tensorflow as tf
# 创建常量
ct_1 = tf.constant([[2,3,4]])
ct_2 = tf.constant([[4],[2],[1]])
# 创建矩阵乘法
dot_product = tf.matmul(ct_1, ct_2)
# 创建会话
# 使用with创建，代码执行后自动关闭会话
with tf.Session() as sess:
    # 使用 run 执行操作  
    result = sess.run(dot_product)
    print(result)

import numpy as np
# 0 - 784  含头不含尾
a = np.arange(784)
print(a)
# 转为 28 * 28 的二维数组， 不改变数据
print(a.reshape(28, 28))


# Fetch
# 会话中同时执行多个操作
input_1 = tf.constant(1.0)
input_2 = tf.constant(2.0)
input_3 = tf.constant(3.0)
# 加法
add = tf.add(input_2, input_1)
# 乘法
mul = tf.multiply(input_3, add)
with tf.Session() as sess:
    result = sess.run([add, mul])
    print(result)


# Feed
# 占位符作默认参数，需要时传值
input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)
# 输出乘法
output = tf.multiply(input_1, input_2)
with tf.Session() as sess:
    # feed的数据需要以字典的形式传入
    print(sess.run(output, feed_dict={input_1: [3.0], input_2: [4.0]}))
