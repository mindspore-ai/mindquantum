import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

def get_random_results(N):
    env = gym.make('CartPole-v0')
    env.reset()
    random_episodes = 0
    reward_sum = 0
    results = []
    while random_episodes < N:
        _, reward, done, _ = env.step(np.random.randint(0,2))
        reward_sum += reward
        if done:
            random_episodes += 1
            results.append(reward_sum)
            reward_sum = 0
            env.reset()
    plt.plot(results)
    plt.ylim(0,200)
    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig('result_random.jpg')
    return results

