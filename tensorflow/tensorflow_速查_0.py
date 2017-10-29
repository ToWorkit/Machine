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
