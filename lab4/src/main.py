from matplotlib import pyplot as plt
from my_solver import MySolver
import numpy as np


DSET_PATH = '/home/smtech/prv/wsi/lab4_dataset/cardio_train.csv'


def main():
    solver = MySolver(max_depth=5)
    solver.load_data(path=DSET_PATH, class_label='cardio')

    experiment_range = 8
    depths = range(experiment_range)

    simulation_count = 1
    simulations = {}

    solver.set_depth(5)
    x_train, x_val, x_test, y_train, y_val, y_test = solver.split_data(0.6, 0.2, 0.2, 44)

    # solver.fit(x_train, y_train)
    # acc = solver.evaluate(x_test, y_test)

    # print(f'TEST_SET: \nDepth = 5 \nAccuracy: {acc}')

    for sim in range(simulation_count):


        print('=======================================')
        print('=======================================')
        print(f'========== Simulation no. {sim+1} ==========')

        accuracies = {}
        for depth in depths:

            accuracies[depth] = {}
            print('===================================')
            print(f'============ DEPTH = {depth + 1} ============')

            solver.set_depth(depth)

            solver.fit(x_train, y_train)

            train_acc = solver.evaluate(x_train, y_train)
            val_acc = solver.evaluate(x_val, y_val)
            print(f'TRAINING_SET: Accuracy: {train_acc}')
            print(f'VALIDATION_SET: Accuracy: {val_acc}')

            accuracies[depth]["train"] = train_acc
            accuracies[depth]["valid"] = val_acc

        train_data = [accuracies[i]["train"] for i in range(experiment_range)]
        val_data = [accuracies[i]["valid"] for i in range(experiment_range)]

        simulations[sim] = {}
        simulations[sim]['train_data'] = train_data
        simulations[sim]['val_data'] = val_data
    
    acc_per_depth_train = {}
    acc_per_depth_val = {}

    # for sim in range(simulation_count):
    for depth in range(experiment_range):
        acc_per_depth_train[depth] = [simulations[sim]["train_data"][depth] for sim in range(simulation_count)]
        acc_per_depth_val[depth] = [simulations[sim]["val_data"][depth] for sim in range(simulation_count)]

    train_plot = [np.mean(acc_per_depth_train[depth]) for depth in range(experiment_range)]
    val_plot = [np.mean(acc_per_depth_val[depth]) for depth in range(experiment_range)]

    x = np.linspace(1, experiment_range, experiment_range)

    plt.plot(x, train_plot)
    plt.plot(x, val_plot)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Max. depth')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
