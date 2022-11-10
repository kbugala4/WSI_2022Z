from solver import Solver
from math import exp
import numpy as np
from plotter import show_plots


def f(X):
    [x] = X
    y = pow(x, 4) / 4
    return y


def df(X):
    [x] = X
    dY = np.array([pow(x, 3)])
    return dY


def g(X):
    [x1, x2] = X
    y = 2 - exp(-pow(x1, 2) - pow(x2, 2)) - 0.5*exp(-pow((x1 + 1.5), 2) - pow((x2 - 2), 2))
    return y


def dg(X):
    [x1, x2] = X
    y_dx1 = 2*x1*exp(-pow(x1, 2) - 2*pow(x2, 2)) + (1.5 + x1)*exp(-pow((1.5 + x1), 2) - pow((-2 + x2), 2))
    y_dx2 = 2*x2*exp(-pow(x1, 2) - pow(x2, 2)) + (x2 - 2)*exp(-pow((x1 + 1.5), 2) - pow((x2 - 2), 2))
    dY = np.array([y_dx1, y_dx2])
    return dY


def problem_f(X):
    return f(X), df(X)


def problem_g(X):
    return g(X), dg(X)


class MySolver(Solver):
    def __init__(self, lrn_rate=1.0, max_iter=1000, stop_cond=0.01, decay=0.1):
        """
        lrn_rate : learning rate (initial step size)
        max_iter : max number of iterations
        stop_cond : stop condition (tolerance)
        decay : coefficient of decay weight for step size
        """
        self.lrn_rate = lrn_rate
        self.max_iter = max_iter
        self.stop_cond = stop_cond
        self.decay = decay

    def get_parameters(self):
        params = {
            "Learn_rate": self.lrn_rate,
            "Max_iters": self.max_iter,
            "Stop_condition": self.stop_cond,
            "Decay_weight": self.decay
        }
        return params

    def solve(self, problem, x0, *args, **kwargs):
        curr_lrn_rate = self.lrn_rate
        next_x = x0
        feedback = "Maximum number of iterations reached"
        is_valid = False
        for iter in range(self.max_iter):
            curr_x = next_x
            curr_y, curr_gradient = problem(curr_x)
            if np.all(abs(curr_gradient) < self.stop_cond):
                feedback = "Gradient near zero"
                break
            next_x = curr_x - curr_gradient * curr_lrn_rate
            if np.linalg.norm(next_x - curr_x) < self.stop_cond:
                feedback = "No X progress detected"
                break

            next_y, _ = problem(next_x)
            while next_y > curr_y:
                curr_lrn_rate *= (1 - self.decay)
                next_x = curr_x - curr_gradient * curr_lrn_rate
                next_y, _ = problem(next_x)
                if np.linalg.norm(next_x - curr_x) < self.stop_cond:
                    feedback = "No X progress detected"
                    break
            if iter > 0:
                x_history = np.vstack([x_history, [curr_x]])
                y_history = np.vstack([y_history, [curr_y]])
                gradient_history = np.vstack([gradient_history, [curr_gradient]])
                lrn_rate_history = np.vstack([lrn_rate_history, [curr_lrn_rate]])
            else:
                is_valid = True
                x_history = np.array([curr_x])
                y_history = np.array(curr_y)
                gradient_history = np.array([curr_gradient])
                lrn_rate_history = np.array([self.lrn_rate])
        if is_valid:
            history = {
                "x": x_history,
                "y": y_history,
                "gradient": gradient_history,
                "learning_rate": lrn_rate_history,
                "iters": iter,
                "feedback": feedback,
            }
            return history
        else:
            print('----- Can\'t process. Choose different X0 -----')
            return None


if __name__ == "__main__":
    """
    Setting up hyperparameters' values:
    """
    lrn_rate = 0.2
    max_iter = 10000
    stop_cond = 0.0001
    decay = 0.03

    solver = MySolver(lrn_rate, max_iter, stop_cond, decay)
    param_dict = solver.get_parameters()
    hist = solver.solve(problem_g, [0.8, 2.5])
    if hist is not None:
        print(" X0: {}\n Xmin: {}\n Ymin: {}\n iters: {}\n feedback: {}".format(
            hist["x"][0], hist["x"][-1], hist["y"][-1][0], hist["iters"], hist["feedback"]))
        show_plots(hist, g)
