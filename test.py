import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    obs, rew, done, _ = env.step(1)  # take a random action
    print(obs, rew, done)
env.close()