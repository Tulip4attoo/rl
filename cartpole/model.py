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
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 2], name="actions_")

            self.dense1 = tf.layers.dense(inputs = self.inputs_,
                                    units = 4,
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="dense1")
            self.dense2 = tf.layers.dense(inputs = self.dense1,
                                    units = 20,
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="dense2")
            self.dense3 = tf.layers.dense(inputs = self.dense2,
                                    units = 10,
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="dense3")
            self.output = tf.layers.dense(inputs = self.dense3, 
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    units = 2, 
                                    activation=None)

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
