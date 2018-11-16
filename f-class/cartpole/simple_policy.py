import numpy as np
import gym
import time
import tensorflow as tf
from collections import deque # Ordered collection with ends
from model import DQNetwork
import time

env = gym.make("CartPole-v1")

state = env.reset()
done = False

def run_policy():
    done = False
    state = env.reset()
    while not done:
        if state[2] <= 0:
            new_state, reward, done, _ = env.step(0)
            state = new_state
            env.render()
            time.sleep(0.2)
        else:
            new_state, reward, done, _ = env.step(1)
            state = new_state
            env.render()
            time.sleep(0.2)




