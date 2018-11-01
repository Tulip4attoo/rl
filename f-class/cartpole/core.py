import numpy as np
import gym
import time
import tensorflow as tf
from collections import deque # Ordered collection with ends
from model import DQNetwork

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

env = gym.make("CartPole-v1")

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

DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# PART II: GEN MEMORY
print("gen memory")

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        return [self.buffer[i] for i in index]

memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
    action = np.random.randint(0, action_size)
    new_state, reward, done, _ = env.step(action)
    memory.add((state, action, reward, new_state, done))
    if done:
        state = env.reset()
    else:
        state = new_state


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = np.random.randint(0, action_size)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability


# PART III: TRAIN AGENT

tf.reset_default_graph()

print("train agent")
# Saver will help us to save our model
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    
    # Initialize the decay rate (that will use to reduce epsilon) 
    decay_step = 0

    for episode in range(total_episodes):
        step = 0
        episode_rewards = []
        state = env.reset()

        while step < max_steps:
            step += 1
            decay_step +=1

            # Predict the action to take and take it
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
            new_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            if done:
                # Set step = max_steps to end the episode
                step = max_steps
                total_reward = np.sum(episode_rewards)
                print('Episode: {}'.format(episode),
                            'Total reward: {}'.format(total_reward),
                            'Training loss: {:.4f}'.format(loss),
                            'Explore P: {:.4f}'.format(explore_probability))
            else:
                new_state, reward, done, _ = env.step(action)
                memory.add((state, action, reward, new_state, done))
                state = new_state

            ### LEARNING PART            
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([np.array((1,0)) if i == 0 else np.array((0,1)) for i in actions_mb])
            rewards_mb = np.array([each[2] for each in batch]) 
            next_states_mb = np.array([each[3] for each in batch])
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []
            # Get Q values for next_state 
            Qs_next_state = sess.run(DQNetwork.output, \
                    feed_dict = {DQNetwork.inputs_: next_states_mb})
            
            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]
                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + discount_rate * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)
            targets_mb = np.array([each for each in target_Qs_batch])
            loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                feed_dict={DQNetwork.inputs_: states_mb,
                                            DQNetwork.target_Q: targets_mb,
                                            DQNetwork.actions_: actions_mb})

        if episode % 500 == 0:
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model Saved")
