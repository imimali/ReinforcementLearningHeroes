'''
    created on 10 March 2019
    
    @author: Gergely
'''
import numpy as np


def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for probability, next_state, reward, done in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])

    return action_values


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iter=1e9):
    evaluation_iters = 1
    V = np.zeros(environment.nS)

    for i in range(int(max_iter)):
        delta = 0

        for state in range(environment.nS):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for state_prob, next_state, reward, done in environment.P[state][action]:
                    v += action_prob * state_prob * (reward + discount_factor * V[next_state])

            delta = max(delta, abs(V[state] - V))
            V[state] = v

        evaluation_iters += 1
        if delta < theta:
            print("policy evaluated in", evaluation_iters, "steps")
            return V


def policy_iteration(environment, discount_factor=1.0, max_iters=1e9):
    policy = np.ones((environment.nS, environment.nA)) / environment.nA
    evaluated_policies = 1
    for i in range(int(max_iters)):
        stable_policy = True
        V = policy_evaluation(policy, environment, discount_factor=discount_factor)

        for state in environment.nS:
            current_action = np.argmax(policy[state])
            action_values = one_step_lookahead(environment, state, V, discount_factor)
            best_action = np.argmax(action_values)
            if current_action != best_action:
                stable_policy = False
            policy[state] = np.eye(environment.nA)[best_action]
        evaluated_policies += 1
        if stable_policy:
            print("evaluated", evaluated_policies, "policies")
            return policy, V


def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    V = np.zeros(environment.nS)
    for i in range(int(max_iterations)):
        delta = 0
        for state in environment.nS:
            action_values = one_step_lookahead(environment, state, V, discount_factor)

            best_action_value = np.max(action_values)

            delta = max(delta, abs(V[state] - best_action_value))

            V[state] = best_action_value
        if delta < theta:
            print("value iteration converged after", i, "steps")
            break

    policy = np.zeros((environment.nS, environment.nA))
    for state in range(environment.nS):
        action_values = one_step_lookahead(environment, state, V, discount_factor)
        best_action = np.argmax(action_values)
        policy[state][best_action] = 1.0
    return policy, V
