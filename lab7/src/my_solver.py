import pandas as pd
from math import exp, pi, sqrt
from solver import Solver
import numpy as np
import operator
from sklearn.model_selection import train_test_split
from copy import copy
# from sklearn.model_selection import train_test_split


class MySolver(Solver):
    def __init__(self, df=None):
        self.df = df

    def load(self, file, class_label, non_discrete=[]):
        self.non_discrete = non_discrete
        df = pd.read_csv(file, delimiter=";")
        self.df = df.drop('id', axis=1)
        self.df_len = len(df)
        self.atts = self.df.drop(class_label, axis=1).columns
        self.classes = np.unique(df[class_label])
        self.class_label = class_label
        self.X_set = self.df.drop(self.class_label, axis=1)
        self.Y_set = self.df[self.class_label]

    def shuffle_df(self):
        self.df = self.df.sample(frac = 1)
        self.X_set = self.df.drop(self.class_label, axis=1)
        self.Y_set = self.df[self.class_label]

    def split_data(self, split_rule):
        """
        A function to split data between train-set, validation-set and test-set
        """
        train_size, validation_size, test_size = split_rule
        X_train, X, Y_train, Y = train_test_split(self.X_set, self.Y_set, test_size=train_size)
        val_test_ratio = validation_size / (validation_size + test_size)
        X_val, X_test, Y_val, Y_test = train_test_split(X, Y, test_size=val_test_ratio)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def split_cross_validation(self,  fold):
        X_samples = []
        Y_samples = []
        sample_size = self.df_len // fold
        self.shuffle_df()

        for i in range(fold):
            X_samples.append(self.X_set[i*sample_size : (i+1)*sample_size])
            Y_samples.append(self.Y_set[i*sample_size : (i+1)*sample_size])

        return X_samples, Y_samples

    def calc_distribution(self, non_discrete_atts, dataset, class_label):
        distribution = {}
        class_distribution = {}

        set_size = len(dataset)
        for class_val in self.classes:
            distribution[class_val] = {}
            class_subset = dataset.query(f'{class_label} == {class_val}')
            class_distribution[class_val] = len(class_subset) / set_size 
            for attribute in self.atts:
                distribution[class_val][attribute] = {}
                att_values = class_subset[attribute]

                if attribute in non_discrete_atts:
                    mean = np.mean(att_values)
                    std_dev = np.std(att_values)
                    distribution[class_val][attribute]['mean'] = mean
                    distribution[class_val][attribute]['std_dev'] = std_dev
                else:
                    unique_values = np.unique(att_values)
                    subset_count = len(att_values)
                    for val in unique_values:
                        val_count = len(class_subset.query(f'{attribute} == {val}'))
                        distribution[class_val][attribute][val] = val_count / subset_count
        return distribution, class_distribution

    def fit(self, X, y):
        train_set = pd.concat([X, y], axis=1)
        self.dist, self.class_dist = self.calc_distribution(self.non_discrete, train_set, self.class_label)

    def predict(self, X):
        predictions = []
        for _, record in X.iterrows():
            predictions.append(self.predict_record(record))
        return predictions

    def predict_record(self, record):
        class_prob = {}
        for class_val in self.classes:
            class_prob[class_val] = self.class_dist[class_val]

            for attribute in self.atts:
                att_value = record[attribute]
                if attribute in self.non_discrete:
                    mean = self.dist[class_val][attribute]['mean']
                    std_dev = self.dist[class_val][attribute]['std_dev']
                    cond_prob = (1 / (std_dev*sqrt(2*pi))) * exp(
                                -((att_value - mean) ** 2 / (2*std_dev**2)))
                else:
                    cond_prob = self.dist[class_val][attribute][att_value]

                class_prob[class_val] *= cond_prob

        record_class = max(class_prob.items(), key=operator.itemgetter(1))[0]

        return record_class

    def evaluate_train_valid(self, X_train, Y_train, X_valid, Y_valid):
        self.fit(X_train, Y_train)

        train_prediction = self.predict(X_train)
        valid_prediction = self.predict(X_valid)

        train_acc = self.calculate_accuracy(train_prediction, Y_train)
        valid_acc = self.calculate_accuracy(valid_prediction, Y_valid)
        return train_acc, valid_acc

    def evaluate_cross_validation(self, X_samples, Y_samples):
        train_accs = []
        valid_accs = []
        for sample in range(len(X_samples)):
            X_train = copy(X_samples)
            Y_train = copy(Y_samples)
            X_valid = X_train.pop(sample)
            Y_valid = Y_train.pop(sample)
            X_train = pd.concat(X_train, axis=0)
            Y_train = pd.concat(Y_train, axis=0)

            train_acc, valid_acc = self.evaluate_train_valid(X_train, Y_train, X_valid, Y_valid)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)

        return train_accs, valid_accs

    def calculate_accuracy(self, predictions, labels):
        is_correct = predictions == labels
        correct = is_correct.value_counts()[True]
        correct_rate = correct / len(labels)
        return correct_rate
