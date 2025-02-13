import numpy as np
from collections import defaultdict

class Agent_Q_learning():

    def __init__(self, alpha = 0.5, gamma = 1, eps = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = defaultdict(int)
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
        if R == 0:
            next_Q = 0
        else:
            next_Q = self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (R + self.gamma * next_Q - self.Q[state, action])