import numpy as np
import gym
import time

env = gym.make("Taxi-v2")

FILE_SAVE = "q_table.npy"
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode

q_table = np.load(FILE_SAVE)
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    #print("****************************************************")
    #print("EPISODE ", episode)

    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info = env.step(action)
        env.render()
        total_rewards += reward
        print(total_rewards)
        time.sleep(0.1)
        
        if done:
            rewards.append(total_rewards)
            #print ("Score", total_rewards)
            break
        state = new_state
env.close()
print ("Score over time: " +  str(sum(rewards)/total_test_episodes))
