import input_data
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一个占位符x
x = tf.placeholder(tf.float32, [None, 784])  # 张量的形状是[None, 784]，None表第一个维度任意

# 定义变量W,b,是可以被修改的张量，用来存放机器学习模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现模型, y是预测分布
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型，y_是实际分布
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉嫡，cost function

# 使用梯度下降来降低cost，学习速率为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化已经创建的变量
init = tf.global_variables_initializer()

# 在一个Session中启动模型，并初始化变量
sess = tf.Session()
sess.run(init)

# 训练模型，运行1000次，每次随机抽取100个
for i in range(1, 1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 验证正确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
