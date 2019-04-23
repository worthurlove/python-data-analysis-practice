import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

#数据集输入输出节点设置

INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络的参数
LAYER1_NODE = 500#隐藏节点数，本模型使用单隐层

BATCH_SIZE = 100#一轮训练中数据个数

LEARNING_RATE_BASE = 0.8#基础学习率

LEARNING_RATE_DECAY = 0.99#学习率的衰减率

REGULARIZATION_RATE = 0.0001#模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000#训练轮数
MOVING_AVERAGE_DECAY = 0.99#滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1,weights2,biases2):

    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:

        #计算隐藏层的前向传播结果，使用RELU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)#bases1为激活函数中边权重之外的常数系数

        #计算输出层的向前传播结果
        return tf.matmul(layer1,weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) 
        + avg_class.average(biases1))

        return tf.matmul(layer1,avg_class.average(weights2) + avg_class.average(biases2))


#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')#placeholder机制用于提供数据，意思就是先占一个坑，说我有一个数组在这

    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #生成输入层到隐藏层的边权重
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1)
    )
    #生成偏置项
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    #生成隐藏层到输出层的边权重
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1)
    )
    #生成偏置项
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算神经网前项传播得结果，不使用滑动平均

    y = inference(x,None,weights1,biases1,weights2,biases2)

    #定义训练轮数
    global_step = tf.Variable(0,trainable=False)

    #滑动平均相关参数
    varible_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    varible_averages_op = varible_averages.apply(tf.trainable_variables())

    average_y = inference(x,varible_averages,weights1,biases1,weights2,biases2)


    #计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))

    #计算本轮所有训练样本的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #计算模型的正则化损失，一般只计算神经网络边上权重，不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    

    #梯度下降优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step=global_step)


    #神经网络模型训练时，即需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step,varible_averages_op]):
        train_op = tf.no_op(name='train')



    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据

        validate_feed = {x:mnist.validation.images,
                        y:mnist.validation.lables
                        }

        #准备测试数据，作为模型优劣的评价标准
        test_feed = {x:mnist.test.images,
                    y:mnist.test.lables
                    }

        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print('after {} steps,validation is {}'.format(i,validate_acc))

            #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)

            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)

        print('after {} steps,test accuracy is {}'.format(TRAINING_STEPS,test_acc))


    #主程序入口

def main(argv=None):
    mnist = input_data.read_data_sets('E:/course-work/DataMining/DeepLearning/MNIST_data',one_hot='True')
    train(mnist)

if __name__ == '__main__':
    main()