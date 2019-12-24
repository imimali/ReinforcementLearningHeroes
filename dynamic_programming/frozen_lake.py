'''
    created on 10 March 2019
    
    @author: Gergely
'''
import gym
import numpy as np
from dynamic_programming.dp import value_iteration, policy_iteration

action_mappings = {
    0: '\u2191',
    1: '\u2191',
    2: '\u2192',
    3: '\u2193'
}

n_episodes = 1000


def play_episodes(environement, n_episodes, policy):
    wins = 0
    total_reward = 0

    for episode in range(n_episodes):
        terminated = False
        state = environement.reset()

        while not terminated:
            action = np.argmax(policy[state])

            next_state, reward, terminated, info = environement.step(action)
            # environement.render()

            total_reward += reward

            state = next_state
            if terminated and reward == 1.0:
                wins += 1
    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward


solvers = [
    ('Policy Iteration', policy_iteration),
    ('Value Iteration', value_iteration)
]

for iteration_name, iteration_func in solvers:
    environment = gym.make('FrozenLake8x8-v0')
    # environment = gym.make('Taxi-v2')
    print('environement done')

    print('running %s ...' % iteration_name)
    policy, V = iteration_func(environment.env)
    print('done')

    print('\n Final policy derived using {%s}:' % iteration_name)
    print(''.join([action_mappings[action] for action in np.argmax(policy, axis=1)]))
    wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)

    print('%s :: number of wins over %d episodes = %d' % (iteration_name, n_episodes, wins))
    print('%s :: average reward over %d episodes = %.2f \n' % (iteration_name, n_episodes, average_reward))
