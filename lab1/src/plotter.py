# from xml.etree.ElementTree import QName
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def show_plots(history, f):
    X = history["x"]
    x0 = X[0]
    Y = history["y"]
    iters = history["iters"]
    rate = history["learning_rate"][0]
    problem_dim = X.shape[1]
    if problem_dim == 1:
        x_args = np.linspace(X[0], X[-1], 101)
        y_vals = []
        for x_arg in x_args:
            y_vals.append(f(x_arg))
        fig_2D_x = plt.figure()
        plt.scatter(X[0], Y[0], marker='x', color="g", s=10, label='x0')
        plt.scatter(X[1:-2], Y[1:-2], marker='o', color="b", s=1, label='x_i_step')
        plt.scatter(X[-1], Y[-1], marker='x', color="r", s=10, label='solution')
        plt.plot(x_args, y_vals, color="gray", alpha=0.2, label='f(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()

        fig_2D_fx = plt.figure()
        plt.scatter(0, Y[0], marker='x', color='g', s=10, label='f(x0)')
        plt.scatter(range(len(Y[1:-2])), Y[1:-2], marker='o', color='b', s=1, label='f(x_i)')
        plt.scatter(len(X), Y[-1], marker='x', color='r', s=10, label='solution')
        plt.ylim(np.min(Y)/10, np.max(Y)*10)
        plt.xlabel('i')
        plt.ylabel('f(x_i)')
        plt.legend()
        plt.yscale('log')

        # plt.show()

    elif problem_dim == 2:
        fig_3D_x, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(-5, 5, 101)
        y = np.linspace(-5, 5, 101)
        mesh_X, mesh_Y = np.meshgrid(x, y)
        Z = np.zeros(mesh_X.shape)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i,j] = f([x[i], y[j]])
        surf = ax.plot_surface(mesh_Y, mesh_X, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.2)
        ax.scatter(X[0,0], X[0,1], Y[0], color='g',  marker='x', s=20, label='X0')
        ax.scatter(X[1:-2,0], X[1:-2, 1], Y[1:-2], color='b', s=1, label='X_i_step')
        ax.scatter(X[-1,0], X[-1,1], Y[-1], color='r',  marker='x', s=15, label='minimum')
        ax.legend()
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('g(x1,x2)')
        ax.set_zlim(np.min(Z) - 1/5, np.max(Z) + 1/5)
        fig_3D_x.colorbar(surf, shrink=0.5, aspect=5)

        fig_3D_gx, ax = plt.subplots(1, 1)
        cp = ax.contourf(mesh_Y, mesh_X, Z, alpha=0.6)

        plt.colorbar(cp, label='g(x1,x2)')
        plt.scatter(X[0,0], X[0,1], marker='x', color='g', s=10, label='X0')
        plt.scatter(X[1:-2,0], X[1:-2,1], color='b', s=1, label='X_i_step')
        plt.scatter(X[-1,0], X[-1,1], color='r',marker='x', s=10, label='solution')
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(min(np.min(X[:,0]-3),-3), max(np.max(X[:,0]+3),2))
        plt.ylim(min(np.min(X[:,1]-3), -2), max(np.max(X[:,1]+3), 3.5))

        plt.show()
