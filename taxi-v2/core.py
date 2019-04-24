import numpy as np
import gym
import time

env = gym.make("Taxi-v2")

state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

total_episodes = 5000         # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
discount_rate = 0.618         # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

for episode in range(total_episodes):
    state = env.reset()
    done = False
    for step in range(max_steps):
        epsilon = min(min_epsilon, epsilon*decay_rate)
        # check if we exploit or not
        if np.random.rand() < epsilon:
            # exploit
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, action_size)
        # get the reward and the next state
        new_state, reward, done, _ = env.step(action)
        # update the q_table via Bellman equation
        update = reward + discount_rate*q_table[new_state].max() - q_table[state,action]
        q_table[state,action] = q_table[state,action] + learning_rate*update
        state = new_state
        if done:
            break

    if episode % 1e3 == 0:
        print("done episode ", episode)

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    #print("****************************************************")
    #print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info = env.step(action)
        env.render()
        total_rewards += reward
        print(total_rewards)
        time.sleep(0.5)
        
        if done:
            rewards.append(total_rewards)
            #print ("Score", total_rewards)
            break
        state = new_state
env.close()
print ("Score over time: " +  str(sum(rewards)/total_test_episodes))
