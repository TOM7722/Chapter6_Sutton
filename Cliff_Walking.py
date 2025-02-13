from xxlimited_35 import error
from Agent import Agent
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class Cliff_Walking():
    def __init__(self, wind = False):
        # origin is defined as left_lower corner. and start from 1
        self.xlimit = 12
        self.ylimit = 4
        self.wind = wind
    def step(self, pos, action):
        x, y = pos
        windX = 0
        windY = 0
        if self.wind:
            axe = np.random.randint(0, 2)
            if axe == 0:
                windX = np.random.randint(-1, 2)
                windY = 0
            else:
                windY = np.random.randint(-1, 2)
                windX = 0

        if action == 'left':
            next_state = max(1, x-1+windX), min(y+windY, self.ylimit)
        elif action == 'right':
            next_state = min(self.xlimit, x+1+windX), min(y+windY, self.ylimit)
        elif action == 'up':
            next_state = min(self.xlimit, x+windX), min(y+1+windY, self.ylimit)
        elif action == 'down':
            next_state = min(self.xlimit, x+windX), max(y-1+windY, 1)
        else:
            raise ValueError

        if x == 0:
            next_state = 1, y
        elif y == 0:
            next_state = x, 1

        if next_state == (12, 1):
            return next_state, 0
        elif next_state[1] == 1 and  next_state[0] != 1:
            next_state = 1, 1
            return next_state, -100
        else:
            return next_state, -1



def game(sarsa, q_learning):
    env = Cliff_Walking()
    EPISODE = 500
    hist = []
    agent = Agent(sarsa = sarsa, q_learning = q_learning)
    for episode in range(EPISODE):
        state = 1, 4
        action = agent.e_greedy(state)
        total_reward = 0
        for i in range(5000):
            next_state, R = env.step(state, action)
            next_action = agent.e_greedy(next_state)
            if agent.sarsa:
                agent.update(R, state, action, next_state, next_action)
            elif agent.q_learning:
                agent.update(R, state, action, next_state, agent.greedy(next_state))
            else:
                raise error
            state = next_state
            action = next_action
            if R == 0 or i == 4999:
                hist.append(total_reward)
                break
            else:
                total_reward += R

        '''state = 1, 4
        action = agent.greedy(state)
        for step in range(5000):
            next_state, R = env.step(state, action)
            if agent.sarsa:
                next_action = agent.e_greedy(next_state)
            elif agent.q_learning:
                next_action = agent.greedy(state)
            else:
                raise error
            agent.update(R, state, action, next_state, next_action)
            state = next_state
            action = next_action
            if R == 0:
                hist.append(total_reward)
                # means next state is terminal
                break
            else:
                total_reward += R'''
    return hist[30:], agent.Q




hist_sarsa = []
hist_q_learning = []
Q_sarsa = defaultdict(lambda: 0)
Q_sarsa_coef = defaultdict(lambda: 0)
Q_q_learning = defaultdict(lambda: 0)
Q_q_learning_coef = defaultdict(lambda: 0)
for experiment in tqdm(range(500)):
    hist_s, Q_s = game(True, False)
    hist_q, Q_q = game(False, True)
    hist_sarsa.append(hist_s)
    hist_q_learning.append(hist_q)
    for key, value in Q_s.items():
        Q_sarsa[key] = (Q_sarsa[key] * Q_sarsa_coef[key] + value) / (Q_sarsa_coef[key] + 1)
        Q_sarsa_coef[key] += 1
    for key, value in Q_q.items():
        Q_q_learning[key] = (Q_q_learning[key] * Q_q_learning_coef[key] + value) / (Q_q_learning_coef[key] + 1)
        Q_q_learning_coef[key] +=1


hist_sarsa = np.mean(hist_sarsa, axis = 0)
hist_q_learning = np.mean(hist_q_learning, axis = 0)

# draw part
plt.style.use('dark_background')
plt.figure(figsize=(10, 10))
plt.title('Sarsa vs Q-learning  in Cliff_Walking', fontsize = 'xx-large')
plt.xlabel('Episodes (averaged over 500 experiments)', fontsize = 'xx-large')
plt.ylabel('Rewards',fontsize = 'xx-large')
plt.plot(hist_sarsa, '-', c = 'red', label = 'sarsa')
plt.plot(hist_q_learning, '-', c = 'blue', label = 'q-learning')
plt.legend(loc = 'best', prop = {'size':12})
plt.show()

agent_sarsa = Agent(Q = Q_sarsa)
agent_q_learning = Agent(Q = Q_q_learning)

agent_sarsa.display_grid(y_max = 4)
agent_q_learning.display_grid(y_max = 4)