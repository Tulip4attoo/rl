import numpy as np
import gym
import time
import tensorflow as tf
from model import DQNetwork

env = gym.make("CartPole-v1")

show = False

if show:
    num_steps = 1
else:
    num_steps = 100

state_size = 4
action_size = env.action_space.n
possible_actions = [0,1]
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 5000        # Total episodes for training
max_steps = 200              # Max possible steps in an episode
batch_size = 64             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
discount_rate = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

DQNetwork = DQNetwork(state_size, action_size, learning_rate)

saver = tf.train.Saver()

with tf.Session() as sess:
    totalScore = 0
    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for i in range(num_steps):
        episode_rewards = 0
        env.reset()
        state = env.reset()
        done = False
        while not done:
            if show:
                env.render()
                time.sleep(0.1)
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1,4))})
            action = np.argmax(Qs)
            action = int(action)
            state, reward, done, _ = env.step(action)
            episode_rewards += reward
        print("Score: ", episode_rewards)
        totalScore += episode_rewards
    print("TOTAL_SCORE", totalScore/100.0)
    env.close()

