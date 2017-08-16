import time
import random
import tensorflow as tf

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Qnet:

    def __init__(self, env):

        self.env = env
        self.num_outputs = env.action_space.n

        # define the network inputs
        self.x = tf.placeholder(tf.float32, [None, env.x_lim, env.y_lim, 2])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_outputs])
        self.a = tf.placeholder(tf.float32, [None, self.num_outputs])
        self.eta = tf.placeholder(tf.float32)

        # reshaping should be unnecessary
        # inpt = tf.reshape(x, [None, env.x_lim, env.y_lim, 2])

        # First convolutional layer: full kernal with 2 to 40 channels
        W_conv1 = weight_variable([env.x_lim, env.y_lim, 2, 40])
        b_conv1 = bias_variable([40])
        a_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)

        # Fully connected layer
        W_fc1 = weight_variable([40, 100])
        b_fc1 = bias_variable([100])
        a_conv1_flat = tf.reshape(a_conv1, [-1, 40])
        a_fc1 = tf.nn.relu(tf.matmul(a_conv1_flat, W_fc1) + b_fc1)

        # Map the 100 features into the action space
        W_fc2 = weight_variable([100, self.num_outputs])
        b_fc2 = bias_variable([self.num_outputs])
        self.y = tf.matmul(a_fc1, W_fc2) + b_fc2

        # mask output
        y_mask = self.y * self.a

        # construct a training step
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - y_mask), axis=1))
        self.train_step = tf.train.GradientDescentOptimizer(self.eta).minimize(mse)

        # construct a session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def propagate(self, state):
        return self.sess.run(self.y, feed_dict={self.x: state})
