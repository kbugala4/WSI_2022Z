import numpy as np
import random


class Solver():
    def __init__(self, alpha, gamma, epsilon):
        """
        alpha - learning rate from range (0, 1]
        gamma - discount factor from range [0, 1]
        epsilon - q_table threshold from range [0, 1]
        """
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.is_finished = False
        self.q_table = None

    def load_env(self, env):
        self.env = env
        self.current_state, info = env.reset()
        self.obs_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.initialize_qtable()

    def initialize_qtable(self):
        self.q_table = np.zeros([self.obs_size, self.action_size])

    def update_qtable(self, curr_state, action, step_score, next_state):
        q_value = self.q_table[curr_state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = q_value + self.alpha * (
                    step_score + self.gamma*next_max - q_value)
        self.q_table[self.current_state, action] = new_value

    def explore(self):
        if random.random() > 1 - self.eps:
            action = np.argmax(self.q_table[self.current_state])  # select best action for current state (from q_table)
        else:
            action = self.env.action_space.sample()  # selects random action for current state
        return action

    def make_step(self, action):
        next_state, step_score, terminated, truncated, _ = self.env.step(action)
        is_finished = terminated or truncated
        return step_score, next_state, is_finished

    def choose_action(self, explore=False, q_table=False):
        if explore:
            if random.random() > 1 - self.eps:
                action = np.argmax(self.q_table[self.current_state])  # select best action for current state (from q_table)
            else:
                action = self.env.action_space.sample()  # selects random action for current state
            return action

        if q_table:
            action = np.argmax(self.q_table[self.current_state])
        else:
            action = self.env.action_space.sample()

        return action

    def train_qtable_iteration(self):
            action = self.choose_action(explore=True)
            step_score, next_state, is_finished = self.make_step(action)
            self.update_qtable(self.current_state, action, step_score, next_state)
            self.current_state = next_state
            return is_finished

    def train_qtable(self, iterations):
        for iter in range(iterations):
            is_finished = False
            self.current_state, info = self.env.reset()
            while not is_finished:
                is_finished = self.train_qtable_iteration()
            if iter % 100 == 0:
                print(f'learning iteration: {iter}')

    def solve(self, use_qtable=True):
        self.current_state, info = self.env.reset()
        is_finished = False
        episode_score = 0

        steps = 0
        while not is_finished:
            print(f'\n-------------------------\nSTEP: {steps+1}')
            action = self.choose_action(q_table=use_qtable)   # select best or random action for current state (from q_table)
            step_score, next_state, is_finished = self.make_step(action)
            self.current_state = next_state
            print(f'Action: {action}\nScore: {step_score}\n')
            episode_score += step_score
            steps += 1
        print(f'Is finished after {steps} steps')
        return episode_score, steps

    def evaluate(self, ev_points_count, max_iterations, episodes_count):
        learn_iters = max_iterations/ev_points_count
        ev_points = np.array(range(ev_points_count + 1)) * learn_iters
        av_scores = []
        av_steps = []

        # Episode 0 (random)
        for episode in episodes_count:
            episode_score, steps = self.solve(use_qtable=False)
            stop_scores += episode_score
            stop_steps += steps
        av_scores.append(stop_scores / episodes_count)
        av_steps.append(stop_steps / episodes_count)

        # Episodes after learning
        for stop in ev_points_count:
            self.train_qtable(learn_iters)
            stop_scores = 0
            stop_steps = 0
            for episode in episodes_count:
                episode_score, steps = self.solve(use_qtable=True)
                stop_scores += episode_score
                stop_steps += steps
            av_scores.append(stop_scores / episodes_count)
            av_steps.append(stop_steps / episodes_count)

        return ev_points, av_scores, av_steps
