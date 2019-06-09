'''
    created on 09 June 2019
    
    @author: Gergely
'''
import random


def run_game(env, policy, display=True, should_return=True):
    env.reset()
    episode = []
    done = False

    while not done:
        s = env.env.s

        if display:
            env.render()
        timestep = [s]
        action = policy[s]
        state, reward, done, info = env.step(action)
        timestep.append(action)
        timestep.append(reward)
        episode.append(timestep)

    if should_return:
        return episode


def argmaxQ(Q, s):
    '''
    what the fuck
    :param Q:
    :param s:
    :return:
    '''
    Q_list = list(map(lambda x: x[1], Q[s].items()))
    indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
    max_Q = random.choice(indices)
    return max_Q


def greedy_policy(Q):
    policy = {}
    for state in Q:
        policy[state] = argmaxQ(Q, state)
    return policy


def field_list(env):
    l = []
    for row in list(map(lambda x: list([str(y)[-2] for y in x]), list(env.env.desc))):
        for field in row:
            l.append(field)
    return l


def create_state_action_dictionary(env, policy):
    Q = {}
    fields = field_list(env)
    for key in policy:
        Q[key] = {a: 0.0 for a in range(env.action_space.n)}

    return Q


def test_policy(policy, env):
    wins = 0
    r = 1000
    for i in range(r):
        w = run_game(env, policy, display=False)[-1][-1]
        if w == 1:
            wins += 1
    return wins / r


