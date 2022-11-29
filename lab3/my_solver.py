import numpy as np
from math import inf
from random import choice
import copy
from itertools import combinations
import sys

_games_dir_path = "C:/Users/kbuga/Desktop/Studia Kacper/sem7/WSI/two-player-games"
sys.path.append(_games_dir_path)
from two_player_games.games.Pick import Pick


def heuristic_function(node, curr_depth, is_max_computing, initial_depth):
    """ Heuristic funtion for a given node """

    # If terminal node:
    if node.is_finished():
        penalty = initial_depth - curr_depth
        winner = node.get_winner()

        if not winner:              # It's a draw
            return 0
        elif winner.char == '1':    # -MAX- wins
            return 2**(12 - penalty)
        else:                       # -MIN- wins
            return - 2**(12 - penalty)

    # If player has picked at least n-1 numbers (chance to win in 1 move):
    elif len(node.current_player_numbers) >= node.n - 1:
        def calc_value(player_numbers):
            number_found_factor = 2
            combs_of_one_missing = list(
                combinations(player_numbers, node.n - 1))
            numbers_to_win = set()
            for comb in combs_of_one_missing:
                comb_sum = sum(comb)
                number_to_win = node.aim_value - comb_sum
                if node.max_number >= number_to_win > 0:
                    if (number_to_win > 0) and number_to_win not in node.selected_numbers:
                        numbers_to_win.add(number_to_win)
            return len(numbers_to_win) ** number_found_factor

        value = calc_value(node.current_player_numbers)
        value -= calc_value(node.other_player_numbers)

        if is_max_computing:
            return value
        else:
            return value * (-1)

    # If player has picked less than n-1 numbers (no chance to win in 1 move):
    else:
        return 0


class GameSolver(object):
    def __init__(self, game, heur_val_fun, depth_min, depth_max):
        self.game = game
        self.heur_val_fun = heur_val_fun
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.computed_nodes_this_round = 0
        self.nodes_per_round_max = []
        self.nodes_per_round_min = []

        # 'is_maximizing_move = True' means that -MAX- starts
        # 'is_maximizing_move = False' means that -MIN- starts
        self.is_maximizing_move = True

    def alphabeta(self, node, curr_depth, is_maximizing_player, alpha=-inf, beta=inf):
        """
        Alpha-beta minimax algorithm with default: alpha = -inf, beta = inf.
        Function is recursive, is_maximizing_player value is given as attribute
        (whether computing for -MAX- or -MIN- move)
        """

        # Terminal node
        if curr_depth == 0 or node.is_finished():
            self.computed_nodes_this_round += 1
            if self.is_maximizing_move:
                initial_depth = self.depth_max
            else:
                initial_depth = self.depth_min
            return (self.heur_val_fun(
                    node, curr_depth, self.is_maximizing_move,
                    initial_depth), None)

        computed_nodes = 0

        values = []
        proper_moves = node.get_moves()

        # If this node's current player is -MIN-
        if is_maximizing_player:
            value = -inf
            for move in proper_moves:
                child_node = copy.deepcopy(node)
                child_node = child_node.make_move(move)
                minimax_val, __ = self.alphabeta(
                    child_node, curr_depth-1, not is_maximizing_player, alpha, beta)
                value = max(value, minimax_val)
                alpha = max(alpha, value)
                values.append(alpha)

                computed_nodes += 1
                if value >= beta:
                    break  # beta cutoff

            # If several moves produces same optimal value, choose random
            best_moves = np.argwhere(
                values == np.amax(values)).flatten().tolist()
            best_val = np.max(values)

        # If this node's current player is -MIN-
        else:
            value = inf
            for move in proper_moves:
                child_node = copy.deepcopy(node)
                child_node = child_node.make_move(move)
                minimax_val, __ = self.alphabeta(
                    child_node, curr_depth-1, not is_maximizing_player, alpha, beta)
                value = min(value, minimax_val)
                beta = min(beta, value)
                values.append(beta)

                computed_nodes += 1
                if alpha >= value:
                    break  # alpha cutoff

            best_moves = np.argwhere(
                values == np.amin(values)).flatten().tolist()
            best_val = np.min(values)

        self.computed_nodes_this_round += computed_nodes

        # Return best move and value it gives
        move_to_make = proper_moves[choice(best_moves)]
        return best_val, move_to_make

    def player_move(self, depth):
        """
        A function that performs a single move-round of a player
        with given depth, using alphabeta minimax algorithm.

        Current player-to-move stored in var self.is_maximizing_move
        """
        value, selected_move = self.alphabeta(
            self.game.state, depth, self.is_maximizing_move)

        print('\nSelected number: {}'.format(selected_move.number))
        print('Current value: {}'.format(value))

        if self.is_maximizing_move:
            self.nodes_per_round_max.append(self.computed_nodes_this_round)
        else:
            self.nodes_per_round_min.append(self.computed_nodes_this_round)

        self.game.state = self.game.state.make_move(selected_move)

        self.computed_nodes_this_round = 0

    def play_once(self):
        """
        A funtion that simulates a game with 2 players.
        Each player makes one move per round,
        maximizing player desired to start the game.
        """
        curr_round = 1

        # -MAX- player starts the game
        while not self.game.state.is_finished():
            print('\n========= Round no.{} =========\n'.format(curr_round))
            print('========= Player -MAX- to move =========')
            print(self.game.state)

            # -MAX- player moves now
            self.player_move(self.depth_max)
            self.is_maximizing_move = not self.is_maximizing_move

            if not self.game.state.is_finished():
                # -MIN- player moves now
                print('========= Player -MIN- to move =========')
                print(self.game.state)
                self.player_move(self.depth_min)
                self.is_maximizing_move = not self.is_maximizing_move

            curr_round += 1

        print('\n\n========= Game finished =========\n')

        winner = self.game.state.get_winner()
        if winner:
            winner = int(winner.char)
            print('Game result: player -{}- wins'.format(winner))
        else:
            print('Game result: -draw-')
            winner = 0

        print('Game stopped at node: \n{}'.format(self.game.state))

        nodes_per_round = '\nComputed nodes per round:' + \
                          '\nmaximizing player: {}'.format(
                              self.nodes_per_round_max) + \
                          '\nother player: {}'.format(
                              self.nodes_per_round_min)
        print(nodes_per_round)

        nodes_summed = '\nComputed nodes summed:' + \
                       '\nmaximizing player: {}'.format(
                           sum(self.nodes_per_round_max)) + \
                       '\nother player: {}'.format(
                            sum(self.nodes_per_round_min))
        print(nodes_summed)

        return (winner, sum(self.nodes_per_round_min),
                sum(self.nodes_per_round_max))


if __name__ == '__main__':
    depth_min_player = 3
    depth_max_player = 3

    number_of_games = 20 
    history = np.zeros((number_of_games, 3))

    for counter in range(number_of_games):
        game = GameSolver(Pick(), heuristic_function,
                          depth_min_player, depth_max_player)
        results = game.play_once()
        history[counter] = results

    print('\n========= Statistics =========')
    print('Depth: Maximizing player: {}, Other player: {}'.format(
        depth_max_player, depth_min_player))

    draws = (history[:, 0] == 0).sum()
    max_wins = (history[:, 0] == 1).sum()
    min_wins = (history[:, 0] == 2).sum()

    max_info = '\n-MAX- wins -{}- times'.format(max_wins)
    min_info = '\n-MIN- wins -{}- times'.format(min_wins)
    draw_info = '\ndraws -{}- times'.format(draws)
    print(max_info + min_info + draw_info)

    mean_info = '\nMean computed nodes: -MAX-: {}, -MIN-: {}\n'.format(
        np.mean(history[:, 2]), np.mean(history[:, 1]))

    print(mean_info)
