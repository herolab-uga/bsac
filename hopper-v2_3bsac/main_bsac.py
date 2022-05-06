'''
Author: Qin Yang
05/06/2022
'''

import gym
import numpy as np
from bsac_torch import Agent
from gym import wrappers
import mujoco_py
from rl_plotter.logger import Logger, CustomLogger


if __name__ == '__main__':
    env = gym.make('Hopper-v2')

    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])

    n_games = 10000

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    logger = Logger(log_dir = '/custom_logger/Hopper-v2', exp_name = 'Hopper-v2', env_name = 'myenv', seed = 0)
    custom_logger = logger.new_custom_logger(filename = "loss.csv", fieldnames=["episode_rewards", "average_rewards"])

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'episode rewards %.1f' % score, 'average rewards %.1f' % avg_score)

        custom_logger.update([score, avg_score], total_steps = i)


