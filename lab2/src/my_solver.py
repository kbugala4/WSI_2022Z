from solver import Solver
import numpy as np
import random
from plotter import plot_data


class MySolver(Solver):
    def __init__(self, t_max=600, pc=0.85, pm=0.15, size=400):
        self.t_max = t_max
        self.pc = pc
        self.pm = pm
        self.size = size

    def get_parameters(self):
        params = {
            "t_max": self.t_max,
            "pc": self.pc,
            "pm": self.pm,
            "size": self.size
        }
        return params

    def initialize_p0(self):
        """
        Return random decision vectors
        """
        p0 = np.array([[random.getrandbits(1) for _ in range(200)]
                       for _ in range(self.size)])
        return p0

    def find_best(self, P, scores):
        """
        P, scores are np.arrays
        """
        id_max = np.argmax(scores)
        x_best = P[id_max]
        score_best = scores[id_max]
        return x_best, score_best

    def selection(self, P, scores):
        """
        Shifting the score, so there cannot be negative value
        Selecting better scores with higher probability
        """
        scores = scores - np.amin(scores)
        probability = scores/np.amax(scores)
        probability = probability/np.sum(probability)
        ids = np.array([i for i in range(self.size)])
        selected_ids = np.random.choice(ids, self.size, p=probability)
        P_selected = np.array([P[ids[i]] for i in selected_ids])
        return P_selected

    def crossover_mutation(self, P):
        """
        Function, that performs crossover for given population
        and then mutates each chromosome, both with given probability
        """
        num_of_pairs = int(self.size/2)
        pairs = []
        ids = [i for i in range(num_of_pairs)]
        while ids:
            rand1 = ids.pop(np.random.randint(0, len(ids)))
            rand2 = ids.pop(np.random.randint(0, len(ids)))
            pair = rand1, rand2
            pairs.append(pair)

        P_crossed = P
        for pair in pairs:
            if random.random() < self.pc:
                cross_bit = np.random.randint(1, num_of_pairs)
                tmp = P_crossed[pair[0]]
                for bit in range(cross_bit, 200):
                    P_crossed[pair[0], bit] = P_crossed[pair[1], bit]
                for bit in range(0, cross_bit):
                    P_crossed[pair[1], bit] = tmp[bit]

        P_mutated = P_crossed
        for i in range(self.size):
            for j in range(200):
                if random.random() < self.pm:
                    P_mutated[i, j] = 1 - P_mutated[i, j]

        return P_mutated

    def solve(self, problem, pop0, *args, **kwargs):
        """
        Solves a given problem for single population0,
        returnes globally best vector of decision, score
        and best score per iteration (for plotting)
        """
        def get_scores(P):
            scores = np.array([problem(x) for x in P])
            return scores

        t = 0
        scores = get_scores(pop0)
        x_best_global, score_best_global = self.find_best(pop0, scores)

        P_t = pop0
        P_t_scores = scores

        best_score_per_iter = []
        while t < self.t_max:
            print(f'I: {t} best: {score_best_global} max: {np.max(P_t_scores)}')

            P_t_selected = self.selection(P_t, P_t_scores)
            P_t_mutated = self.crossover_mutation(P_t_selected)

            P_t = P_t_mutated
            P_t_scores = get_scores(P_t)
            x_best_tmp, score_best_tmp = self.find_best(P_t, P_t_scores)
            if score_best_tmp > score_best_global:
                x_best_global = x_best_tmp
                score_best_global = score_best_tmp
                print(f'Improvement! Score: {score_best_global}')

            best_score_per_iter.append(score_best_tmp)
            t += 1
        return x_best_global, score_best_global, np.array(best_score_per_iter)


class Simulation():
    def __init__(self):
        self.time = 200
        self.v0 = 0.0
        self.g = -0.09
        self.F = 45.0  # a = F/mass
        self.height0 = 200.0

    def o_fun(self, X):
        """
        Triggers the objective function and returns score
        for given binary decisions
        """
        decisions = np.array(X)
        gas = np.count_nonzero(decisions)
        result = self.simulate(decisions, gas)
        score = self.calculate_score(gas, **result)
        return score

    def simulate(self, X, gas):
        """
        Simulation of a landing rocket, returns results such as:
        'height_arr' - height per step of the simulation
        'v_arr' - velocity per step of the simulation
        --------------------
        Parameters:
        'X' is a set of decisions, array of:
        1 - motors ON
        0 - motors OFF

        'gas' is mass of gas (count of ones in X)
        """
        height_arr = []
        v_arr = []

        mass0 = 200 + gas

        # Initializing variables
        height = self.height0
        v = self.v0
        mass = mass0

        # Each loop iteration is simulating the rocket's motion
        for time_period in range(X.shape[0]):
            if X[time_period]:
                a = self.F/mass
                mass -= 1
                v += a
            v += self.g
            height = height + v

            # Fill arrays with historical simulation variables
            height_arr.append(height)
            v_arr.append(v)

        # Saving arrays to dictionary type of data
        result = {
            "heights": np.array(height_arr),
            "velocities": np.array(v_arr)
        }
        return result

    def calculate_score(self, gas, heights, velocities):
        """
        Calculating score for a given simulation and gas units
        'sim_data' is a dictionary of simulation data
        'gas' is number of gas units
        """
        is_landed = 0
        is_crashed = 0
        for step in range(len(heights)):
            if heights[step] < 2:
                if heights[step] < 0:
                    is_crashed = -1000
                    break
                elif abs(velocities[step]) < 2:
                    is_landed = 2000
                    break

        score = is_landed + is_crashed - gas

        return score


if __name__ == "__main__":
    solver = MySolver()
    simulation = Simulation()

    statistics = {}

    simulations_number = 25
    for sol in range(simulations_number):
        pop0 = solver.initialize_p0()
        x, score, iter_progress = solver.solve(simulation.o_fun, pop0)
        statistics[f'sim_{sol}'] = [x, score, iter_progress]

    best_scores = [statistics[f'sim_{sol}'][1] for sol in range(simulations_number)]
    best_id = np.argmax(np.array([best_scores]))
    best_score = best_scores[best_id]
    best_x = statistics[f'sim_{best_id}'][0]
    print(f'best_score: {best_score}')
    print(f'best_x: {best_x}')
    plot_data(statistics, simulations_number)
