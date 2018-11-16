import numpy as np
import gym
import time
import tensorflow as tf
from collections import deque # Ordered collection with ends


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # ok so the network structure is:
            # 3 layers of conv
            # flatten
            # self.action_size layers of dense

            # note that the input size is 84 * 84 * 4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                    filters = 32,
                                    kernel_size = 8,
                                    strides = 4,
                                    padding='valid',
                                    activation = tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="conv1")
            self.batchnorm1 = tf.layers.batch_normalization(inputs = self.conv1,
                                    training = True,
                                    epsilon = 1e-5,
                                    name="batchnorm1")
 
            self.conv2 = tf.layers.conv2d(inputs = self.batchnorm1,
                                    filters = 64,
                                    kernel_size = 4,
                                    strides = 2,
                                    padding='valid',
                                    activation = tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="conv2")
            self.batchnorm2 = tf.layers.batch_normalization(inputs = self.conv2,
                                    training = True,
                                    epsilon = 1e-5,
                                    name="batchnorm2")

            self.conv3 = tf.layers.conv2d(inputs = self.batchnorm2,
                                    filters = 64,
                                    kernel_size = 3,
                                    strides = 2,
                                    padding='valid',
                                    activation = tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="conv3")
            self.batchnorm3 = tf.layers.batch_normalization(inputs = self.conv3,
                                    training = True,
                                    epsilon = 1e-5,
                                    name="batchnorm3")

            self.flatten1 = tf.layers.flatten(inputs = self.batchnorm3,
                                    name="flatten1")

            self.dense1 = tf.layers.dense(inputs = self.flatten1,
                                    units = 512,
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="dense1")
            self.dense2 = tf.layers.dense(inputs = self.dense1,
                                    units = 40,
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="dense2")
            self.output = tf.layers.dense(inputs = self.dense2, 
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    units = self.action_size, 
                                    activation=None)

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
