import matplotlib.pyplot as plt
import numpy as np


def plot_data(statistics, count):
    x_len = statistics["sim_0"][2].shape[0]
    x = np.array([i for i in range(x_len)])

    y_low = np.array([i for i in range(x_len)])
    y_high = np.array([i for i in range(x_len)])
    y_mean = np.array([i for i in range(x_len)])

    y_best =  np.array([[i for i in range(x_len)]for _ in range(count)])
    y_best_low = np.array([i for i in range(x_len)])
    y_best_high = np.array([i for i in range(x_len)])
    y_best_mean = np.array([i for i in range(x_len)], dtype=np.float64)
    for iter in range(x_len):
        best_scores = np.array([statistics[f'sim_{sim}'][2][iter] for sim in range(count)])
        y_low[iter] = np.amin(best_scores)
        y_high[iter] = np.amax(best_scores)
        y_mean[iter] = np.mean(best_scores)
        for sim in range(count):
            if iter == 0:
                y_best[sim][iter] = best_scores[sim]
            elif best_scores[sim] > y_best[sim][iter-1]:
                y_best[sim][iter] = best_scores[sim]
            else:
                y_best[sim][iter] = y_best[sim][iter-1]

        global_best_scores = np.array([y_best[sim][iter] for sim in range(count)])
        y_best_mean[iter] = np.mean(global_best_scores, dtype=np.float64)
        y_best_low[iter] = np.amin(global_best_scores)
        y_best_high[iter] = np.amax(global_best_scores)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.fill_between(x, y_low, y_high, alpha=.5, linewidth=0)
    ax.plot(x, y_mean, linewidth=2, color='r')

    ax2.fill_between(x, y_best_low, y_best_high, alpha=3, linewidth=0)
    ax2.plot(x, y_best_mean, linewidth=2, color='blue')

    ax.set(xlim=(0, x_len),
           ylim=(np.amin(y_low) - 5, np.amax(y_high) + 5))

    ax2.set(xlim=(0, x_len),
            ylim=(np.amin(y_best_low) - 5, np.amax(y_best_high) + 5))

    plt.show()
