import numpy as np
import tensorflow as tf

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
