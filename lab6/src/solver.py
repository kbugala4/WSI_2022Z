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
        self.current_state, info = self.env.reset()
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
            if random.random() < 1 - self.eps:
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
                pass
                # print(f'learning iteration: {iter}')

    def solve(self, use_qtable=True):
        self.current_state, info = self.env.reset()
        is_finished = False
        episode_score = 0

        steps = 0
        illegal_moves = 0
        # print('Solving...')
        while not is_finished:
            # print(f'\n-------------------------\nSTEP: {steps+1}')
            action = self.choose_action(q_table=use_qtable)   # select best or random action for current state (from q_table)
            step_score, next_state, is_finished = self.make_step(action)
            self.current_state = next_state
            # print(f'Action: {action}\nScore: {step_score}')
            episode_score += step_score
            steps += 1
            if step_score == -10:
                illegal_moves += 1
        # print(f'Is finished after {steps} steps')
        return episode_score, steps, illegal_moves

    def evaluate(self, ev_points_count, max_iterations, episodes_count):
        self.current_state, info = self.env.reset()
        
        learn_iters = int(max_iterations/ev_points_count)
        ev_points = np.array(range(1, ev_points_count + 1)) * learn_iters
        av_scores = []
        av_steps = []
        av_illegal_moves = []
        print(f'Learning iterations per ev point: {learn_iters}\nStop points: {ev_points}\nEpisodes per evaluation point: {episodes_count}')

        # Episode 0 (random)
        # stop_scores = 0
        # stop_steps = 0
        # for episode in range(episodes_count):
        #     print(f'Calculating at stop point: 0 (random actions)')
        #     episode_score, steps, illegal_moves = self.solve(use_qtable=False)
        #     stop_scores += episode_score
        #     stop_steps += steps
        # av_scores.append(stop_scores / episodes_count)
        # av_steps.append(stop_steps / episodes_count)

        # Episodes after learning
        for stop in range(ev_points_count):
            print(f'\n=========================\nTraining data (iters {stop*learn_iters} - {(stop + 1)*learn_iters - 1})')
            self.train_qtable(learn_iters)
            stop_scores = 0
            stop_steps = 0
            illegal_moves = 0
            print(f'-----------------\nCalculating average at evaluation stop: {stop}')
            for episode in range(episodes_count):
                episode_score, steps, illegal_move_count = self.solve(use_qtable=True)
                stop_scores += episode_score
                stop_steps += steps
                illegal_moves += illegal_move_count
                print(f'Episode {episode}: steps={steps}, score={episode_score}, ill_moves={illegal_move_count}')
            av_illegal_moves.append(illegal_moves / episodes_count)
            av_scores.append(stop_scores / episodes_count)
            av_steps.append(stop_steps / episodes_count)

        return ev_points, av_scores, av_steps, av_illegal_moves
