import numpy as np
from collections import defaultdict

class Agent():

    def __init__(self, alpha = 0.5, gamma = 1, eps = 0.1, sarsa = True, q_learning = False, Q = None):
        self.sarsa = sarsa
        self.q_learning = q_learning
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = Q if Q is not None else defaultdict(int)
        self.action_space = ['left', 'right', 'up', 'down']


    def greedy(self, state):
        max_action = self.action_space[0]
        max_Q = self.Q[state, max_action]
        for action in self.action_space:
            if self.Q[state, action] > max_Q:
                max_action = action
                max_Q = self.Q[state, action]
        return max_action

    def e_greedy(self, state):
        if np.random.random() > self.eps:
            return self.greedy(state)
        else:
            return np.random.choice(self.action_space)

    def update(self, R, state, action, next_state, next_action):
        next_Q = self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (R + self.gamma * next_Q - self.Q[state, action])

    def display_grid(self, x_max=12, y_max=4):
        print('----------------')
        for y in range(y_max, 0, -1):
            for x in range(1, x_max + 1):
                state = (x, y)
                action = self.greedy(state)
                if action == 'left':
                    print(' < ', end='')
                elif action == 'up':
                    print(' ^ ', end='')
                elif action == 'down':
                    print(' v ', end='')
                else:
                    print(' > ', end='')
            print()  # New line after each row
        print('----------------')