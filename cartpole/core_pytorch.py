import numpy as np
import gym
from collections import deque # Ordered collection with ends
from model_pytorch import DQNetwork
import torch
import torch.optim as optim
import torch.nn as nn

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
discount_rate = 0.95               # Discounting rate, 0.95 is a bit small i guess

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

DQNetwork = DQNetwork(state_size, action_size)


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
With ϵ select a random action, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = np.random.randint(0, action_size)
    else:
        with torch.no_grad():
            t_next = torch.tensor(state.reshape((1, *state.shape))).type('torch.FloatTensor')
            Qs = DQNetwork.forward(t_next)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability



# PART III: TRAIN AGENT

print("train agent")

decay_step = 0
optimizer = optim.RMSprop(DQNetwork.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.MSELoss()

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
        t_next = torch.tensor(next_states_mb).type('torch.FloatTensor')
        with torch.no_grad():
            Qs_next_state = DQNetwork.forward(t_next)

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, len(batch)):
            terminal = dones_mb[i]
            # If we are in a terminal state, only equals reward
            if terminal:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + discount_rate * Qs_next_state[i].max()
                target_Qs_batch.append(target)
        targets_mb = np.array([each for each in target_Qs_batch])

        with torch.no_grad():
            t_action_mb = torch.tensor(actions_mb).type('torch.FloatTensor')
            Qs = torch.sum(torch.mul(DQNetwork.forward(t_next), t_action_mb), dim=1)

        optimizer.zero_grad()
        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        loss = criterion(Qs, torch.tensor(targets_mb, requires_grad=True).type('torch.FloatTensor'))
        loss.backward()
        optimizer.step()

