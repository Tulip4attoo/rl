import numpy as np
import gym
import time
import tensorflow as tf
from collections import deque # Ordered collection with ends
from model import DQNetwork
import cv2
import utils
from memory import Memory

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

env = gym.make("Pong-v0")

stack_size = 4
state_size = [84,84,stack_size]      # stack 4 gray frames into 1 (observer) state
action_size = env.action_space.n
possible_actions = list(range(action_size))
learning_rate =  0.0002      # Alpha (aka learning rate)
is_new_episode = True

### TRAINING HYPERPARAMETERS
total_episodes = 5000          # Total episodes for training
max_steps = 10000              # Max possible steps in an episode
batch_size = 32             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
discount_rate = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100          # Number of experiences the Memory can keep

tf.reset_default_graph()

DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# PART II: GEN MEMORY
print("gen memory")

# class Memory():
#     def __init__(self, max_size):
#         self.buffer = deque(maxlen = max_size)
    
#     def add(self, experience):
#         self.buffer.append(experience)
    
#     def sample(self, batch_size):
#         buffer_size = len(self.buffer)
#         index = np.random.choice(np.arange(buffer_size),
#                                 size = batch_size,
#                                 replace = False)
#         return [self.buffer[i] for i in index]

memory = Memory(max_size = memory_size)
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size) 
for i in range(pretrain_length):
    if i == 0:
        frame = env.reset()
        stacked_frames, stacked_state = utils.stack_frames(stacked_frames, frame, is_new_episode)
    action = np.random.randint(0, action_size)
    new_frame, reward, done, _ = env.step(action)
    stacked_frames, new_stacked_state = utils.stack_frames(stacked_frames, new_frame, is_new_episode)
    memory.add((stacked_state, action, reward, new_stacked_state, done))
    if done:
        frame = env.reset()
        is_new_episode = True
        stacked_frames, stacked_state = utils.stack_frames(stacked_frames, frame, is_new_episode)
        is_new_episode = False
    else:
        stacked_state = new_stacked_state.copy()


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # this function is terrible, should re factor it
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
        frame = env.reset()
        is_new_episode = True
        stacked_frames, stacked_state = utils.stack_frames(stacked_frames, frame, is_new_episode)
        is_new_episode = False

        while step < max_steps:
            step += 1
            decay_step +=1

            # Predict the action to take and take it
            action, explore_probability = predict_action(explore_start, \
                        explore_stop, decay_rate, decay_step, stacked_state, possible_actions)
            new_frame, reward, done, _ = env.step(action)
            stacked_frames, new_stacked_state = utils.stack_frames(stacked_frames, new_frame, is_new_episode)
            episode_rewards.append(reward)
            if done:
                # Set step = max_steps to end the episode
                step = max_steps
                total_reward = np.sum(episode_rewards)
                print('Episode: {}'.format(episode),
                            'Total reward: {}'.format(total_reward),
                            'Training loss: {:.4f}'.format(loss),
                            'Explore P: {:.4f}'.format(explore_probability),
                            "Memory len: {}".format(len(memory.buffer)))
            else:
                new_frame, reward, done, _ = env.step(action)
                stacked_frames, new_stacked_state = utils.stack_frames(stacked_frames, new_frame, is_new_episode)
                memory.add((stacked_state, action, reward, new_stacked_state, done))
                stacked_state = new_stacked_state.copy()

            ### LEARNING PART            
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            actions_mb = utils.encode_list_to_onehot(actions_mb, action_size)
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

        if episode % 50 == 0:
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model Saved")
