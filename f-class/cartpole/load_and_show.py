with tf.Session() as sess:
    totalScore = 0
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    episode_rewards = []
    done = False
    for i in range(1):
        state = env.reset()
        while not done:
            env.render()
            time.sleep(0.1)
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1,4))})
            action = np.argmax(Qs)
            action = int(action)
            state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
        print("Score: ", np.sum(episode_rewards))
        totalScore += np.sum(episode_rewards)
    print("TOTAL_SCORE", totalScore/100.0)
    env.close()

