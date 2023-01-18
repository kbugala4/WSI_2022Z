import numpy as np
import random


class Solver():
    def __init__(self, alpha, gamma, epsilon):
        """
        alpha - learning rate from range (0, 1]
        gamma - discount factor from range [0, 1]
        epsilon - q_table value probability while explorting from range [0, 1]
        """
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.is_finished = False
        self.q_table = None

    def load_env(self, env):
        """
        A method to load the given environment, sets necessary variables
        depending on the observation and action space
        """
        self.env = env
        self.current_state, info = self.env.reset()
        self.obs_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.reset_qtable()

    def reset_qtable(self):
        """
        A function that reset values of Q-values table
        """
        self.q_table = np.zeros([self.obs_size, self.action_size])

    def update_qtable(self, action, step_score, next_state):
        """
        A method to update current Q-values table according to
        the Q-Learning Formula
        """
        q_value = self.q_table[self.current_state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = q_value + self.alpha * (
                    step_score + self.gamma*next_max - q_value)
        self.q_table[self.current_state, action] = new_value

    def make_step(self, action):
        """
        A method to make a single step (action), returns the feedback from
        environment (action reward, following step and boolean if is finished)
        """
        next_state, step_score, terminated, truncated, _ = self.env.step(action)
        is_finished = terminated or truncated
        return step_score, next_state, is_finished

    def choose_best(self):
        """
        A method to select best action according to the Q-Value Table.
        In a case, where multiple actions has as-good values, returns random one
        """
        max_val = np.max(self.q_table[self.current_state])
        best_actions = np.argwhere(self.q_table[self.current_state] == max_val)
        best_actions = best_actions.reshape(1, best_actions.shape[0])[0]
        action = np.random.choice(best_actions)
        return action

    def choose_action(self, explore=False, q_table=False):
        """
        A method to select action. 
        Will return 'best' action or random (explore the space) if 'explore' parameter set to True
        Will return 'best' action, if q_table 'parameter' set to True
        Will return random action (explort the space) if both parameters set to False
        """
        if explore:
            if random.random() > self.eps:
                action = self.choose_best() # select best action for current state (from q_table)
            else:
                action = self.env.action_space.sample()  # selects random action for current state
            return action

        if q_table:
            action = self.choose_best()
        else:
            action = self.env.action_space.sample()

        return action

    def train_qtable_iteration(self):
        """
        Single iteration of training the Q-Value Table, 
        explores ther space, makes the step and updates a Q-Table.
        If current 'game' is finished, returns True
        """
        action = self.choose_action(explore=True)
        step_score, next_state, is_finished = self.make_step(action)
        self.update_qtable(action, step_score, next_state)
        self.current_state = next_state
        return is_finished

    def train_qtable(self, iterations):
        """
        A method to train the agent over given number of 
        'games' (iterations parameter) until the 'game' is finished
        """
        for iter in range(iterations):
            is_finished = False
            self.current_state, info = self.env.reset()
            while not is_finished:
                is_finished = self.train_qtable_iteration()

    def solve(self, state, use_qtable=True, break_at=None):
        """
        A method to solve a single 'game' with given starting state. 
        If use_qtable == True, solves without exploration (uses Q-Value table only)
        """
        self.current_state = state

        episode_score = 0
        steps = 0
        illegal_moves = 0
        is_unsolved = False

        is_finished = False
        while not is_finished:
            action = self.choose_action(q_table=use_qtable)   # select best or random action for current state (from q_table)
            step_score, next_state, is_finished = self.make_step(action)
            self.current_state = next_state
            episode_score += step_score
            steps += 1
            if step_score == -10:
                illegal_moves += 1
            if break_at is not None:
                if steps == break_at:
                    is_unsolved = True
                    break
        return episode_score, steps, illegal_moves, is_unsolved

    def evaluate_random(self, episodes_count):
        """
        A method to evaluate a solver that makes random-action steps.
        """
        stop_scores = 0
        stop_steps = 0
        illegal_moves = 0
        unsolved = 0

        print(f'Calculating for random actions')
        for episode in range(episodes_count):
            state, info = self.env.reset()
            episode_score, steps, illegal_move_count, _ = self.solve(state, use_qtable=False)
            stop_scores += episode_score
            stop_steps += steps
            illegal_moves += illegal_move_count
            
        av_illegal_move = illegal_moves / episodes_count
        av_score = stop_scores / episodes_count
        av_step = stop_steps / episodes_count
        solved_ratio = 1 - (unsolved / episodes_count)      
        print(f'Evaluation point:\nav_score = {av_score}, av_steps = {av_step}, av_illegal_moves = {av_illegal_move}')
    

    def evaluate(self, ev_points_count, max_iterations, episodes_count, breakpoint_ev=200):
        """
        A method to evaluate the solver by solving multiple episodes of the problem
        while learning (given number of training iterations -> evaluation -> training...)
        """
        self.current_state, info = self.env.reset()
        self.reset_qtable()
        
        learn_iters = int(max_iterations/ev_points_count)
        ev_points = np.array(range(1, ev_points_count + 1)) * learn_iters
        av_scores = []
        av_steps = []
        av_illegal_moves = []
        av_solved = []
        print(f'Learning iterations per ev point: {learn_iters}\nStop points: {ev_points}\nEpisodes per evaluation point: {episodes_count}')
        
        # Episode 0 (random)
        stop_scores = 0
        stop_steps = 0
        illegal_moves = 0
        unsolved = 0

        print(f'Calculating for random actions')
        for episode in range(episodes_count):
            state, info = self.env.reset()
            episode_score, steps, illegal_move_count, is_unsolved = self.solve(state, break_at=breakpoint_ev)
            stop_scores += episode_score
            stop_steps += steps
            illegal_moves += illegal_move_count
            if is_unsolved:
                unsolved += 1
        av_illegal_move = illegal_moves / episodes_count
        av_score = stop_scores / episodes_count
        av_step = stop_steps / episodes_count
        solved_ratio = 1 - (unsolved / episodes_count)      
        print(f'Evaluation point:\nav_score = {av_score}, av_steps = {av_step}, av_illegal_moves = {av_illegal_move}, solved_ratio = {solved_ratio}')
    
        # Episodes after learning
        for evaluation_point in range(0, ev_points_count):
            print(f'\n=========================\nTraining data (iters {evaluation_point*learn_iters} - {(evaluation_point + 1)*learn_iters - 1})')
            self.train_qtable(learn_iters)
            stop_scores = 0
            stop_steps = 0
            illegal_moves = 0
            unsolved = 0
            episode = 0
            print(f'-----------------\nCalculating average at evaluation stop: {evaluation_point}')
            for episode in range(episodes_count):
                state, info = self.env.reset()
                episode_score, steps, illegal_move_count, is_unsolved = self.solve(state, use_qtable=True, break_at=breakpoint_ev)
                stop_scores += episode_score
                stop_steps += steps
                illegal_moves += illegal_move_count
                if is_unsolved:
                    unsolved += 1
            
            av_illegal_move = illegal_moves / episodes_count
            av_score = stop_scores / episodes_count
            av_step = stop_steps / episodes_count
            solved_ratio = 1 - (unsolved / episodes_count)

            print(f'Evaluation point:\nav_score = {av_score}, av_steps = {av_step}, av_illegal_moves = {av_illegal_move}, solved_ratio = {solved_ratio}')
            av_illegal_moves.append(av_illegal_move)
            av_scores.append(av_score)
            av_steps.append(av_step)
            av_solved.append(solved_ratio)

        return ev_points, av_scores, av_steps, av_illegal_moves, av_solved
