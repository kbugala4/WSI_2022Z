from solver import Solver
import pandas as pd
import numpy as np
from math import log, inf
from sklearn.model_selection import train_test_split


class DataReader():
    def __init__(self, path):
        """
        Reading data from a given file.
        Writes data into two lists: 
            header - with column names
            rows - each row represents one example

        * deleting column '0' which stands for index *
        """

        df = pd.read_csv(path, delimiter=';')
        self.df = df.drop(columns='id')
        self.data_count = df.shape[0]

    def discretize_att(self, att_name, split_condition):
        """
        A function to discretize an attribute given by name from labels.
        'att_name' - name of an attribute
        'split_condition' - list of values that 'splits' a 1D number line
                        and sets new value with growing trend
        """

        discrete_column = np.array(self.df[att_name])

        for row in range(self.data_count):
            curr_class = 0
            is_discretized = False

            for threshold in split_condition:
                if discrete_column[row] < threshold:
                    discrete_column[row] = curr_class
                    is_discretized = True
                    break
                else:
                    curr_class += 1

            if not is_discretized:
                discrete_column[row] = curr_class
        self.df[att_name] = discrete_column.astype(int)
    
    def discretize_all(self):
        self.discretize_att('age', np.array([6, 18, 26, 45, 60, 80])*365)
        self.discretize_att('weight', [50, 70, 90, 110, 130])
        self.discretize_att('height', [100, 135, 160, 180, 200, 210])
        self.discretize_att('ap_hi', [120, 130, 140, 160, 180])
        self.discretize_att('ap_lo', [80, 85, 90, 100, 110])

    def get_data(self):
        return self.df

    def get_split_data(self, class_label='cardio'):
        """
        A method that returns data sliced into an array of class value
        [classes_set], an array of other attributes [atts_set] and header
        knows as columns' labels
        """
        atts_set = self.df.drop(columns=class_label)
        classes_set = self.df[class_label]
        header = self.df.columns.values
        
        return atts_set, classes_set, header


class MySolver(Solver):
    def __init__(self, max_depth=11):
        print(f"Setting 'max_depth' = {max_depth} for this solver")
        self.max_depth = max_depth

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        parameters = {}
        parameters['depth'] = self.max_depth

        return parameters

    def predict_row(self, tree, atts_row):
        if type(tree) is dict:  # if it is leaf node
            node = next(iter(tree))  # getting first key/feature name of the dictionary
            att_value = atts_row[node]  # value of the feature
            if att_value in tree[node]:  # hecking the feature value in current tree node
                return self.predict_row(tree[node][att_value], atts_row)  #goto next feature
            else:
                return None
        else:
            return tree  # return the value

    def calc_dataset_entropy(self, data_set):
        """
        A method to calculate overall dataset entropy
        """
        classes_set = np.array(data_set.iloc[:, -1])
        classes = np.unique(classes_set)
        classes_set_size = classes_set.shape[0]

        classes_count = np.zeros(classes.shape).astype(int)
        for i in range(classes_count.shape[0]):
            classes_count[i] = sum(classes_set == classes[i])
        classes_count = classes_count.astype(float) / classes_set_size

        entropy = 0
        for occurence in classes_count:
            gain = occurence * log(occurence)
            entropy -= gain
        return entropy

    def calc_subsets_entropy(self, data_set, att_name):
        """
        A method to calculate entropy of a subset based on 
        a given attribute name
        """
        dataset_size = data_set.shape[0]
        att_column = np.array(data_set[att_name])
        att_values = np.unique(att_column)

        subsets_att_entropy = 0
        
        for value in att_values:
            subset = data_set.loc[data_set[att_name] == value]

            subset_size = subset.shape[0]
            subset_entropy = self.calc_dataset_entropy(subset)

            subset_gain = subset_size * subset_entropy
            subsets_att_entropy += subset_gain

        subsets_att_entropy = subsets_att_entropy / dataset_size

        return subsets_att_entropy

    def info_gain(self, data_set, att_name):
        """
        A method to calculate information gain for given attribute
        name
        """
        dataset_entropy = self.calc_dataset_entropy(data_set)
        subsets_entropy = self.calc_subsets_entropy(data_set, att_name)

        info_gain = dataset_entropy - subsets_entropy
        return info_gain

    def find_best_gain(self, data_set):
        """
        Returns name of an attribute that gives best
        information gain
        """
        labels = self.header[0:-1]
        best_gain_val = -inf
        best_gain_label = None

        for label in labels:
            label_info_gain = self.info_gain(data_set, label)
            if label_info_gain > best_gain_val:
                best_gain_val = label_info_gain
                best_gain_label = label

        return best_gain_label

    def load_data(self, path, class_label):
        """
        A method to load data from a given file using DataReader() class
        """
        
        data_reader = DataReader(path)
        data_reader.discretize_all()
        self.data_set = data_reader.get_data()
        self.atts_set, self.classes_set, self.header = data_reader.get_split_data('cardio')
        self.classes = self.classes_set.unique()
        self.att_names = self.header[0:-1]
        self.class_label = self.header[-1]

        atts_count = len(self.header) - 1
        if self.max_depth > atts_count:
            info = f"Error!\n'max_depth' parameter is set to " + \
                f"max_depth={self.max_depth}, while dataset contains only " + \
                f"{atts_count} attribute columns.\nDecreasing 'max_depth' " + \
                f"to maximum value, max_depth={atts_count}"
            print(info)
            self.max_depth = atts_count

    def split_data(self, train_size, validation_size, test_size, random_state):
        """
        A function to split data between train-set, validation-set and test-set
        """

        X_train, X, Y_train, Y = train_test_split(self.atts_set, self.classes_set, test_size=train_size, random_state=random_state)
        val_test_ratio = validation_size / (validation_size + test_size)
        X_val, X_test, Y_val, Y_test = train_test_split(X, Y, test_size=val_test_ratio, random_state=42)
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def make_branch(self, att_name, data_set, curr_depth):
        """
        Creates branches for every unique value of a given attribute,
        returns node dictionary and subset
        """
        node = {}

        for value in data_set[att_name].unique():
            subset = data_set.loc[data_set[att_name] == value]
            subset_size = subset.shape[0]
            is_pure = False

            for class_value in self.classes:
                this_class_data = subset[subset[self.class_label] == class_value]
                this_class_occ = this_class_data.shape[0]
                if this_class_occ == subset_size:
                    node[value] = class_value
                    data_set = data_set[data_set[att_name] != value]
                    is_pure = True

            if not is_pure and curr_depth < self.max_depth:
                node[value] = "unknown_class"

            elif not is_pure:
                most_common = subset[self.class_label].value_counts().idxmax()
                node[value] = most_common

        return node, data_set

    def generate_id3_tree(self, is_initial, root, parent_node, data_set, curr_depth):
        """
        A method to generate an ID3 decision tree with given max_depth 
        (recursive method)
        """
        curr_depth += 1
        if data_set.shape[0] > 0:
            curr_att_name = self.find_best_gain(data_set)
            next_root, data_set = self.make_branch(curr_att_name, data_set, curr_depth)

            if not is_initial:
                root[parent_node] = {}
                root[parent_node][curr_att_name] = next_root
            else:
                root[curr_att_name] = next_root

            for next_node, next_class in next_root.items():
                if str(next_class) == "unknown_class" and curr_depth <= self.max_depth:
                    subset = data_set[data_set[curr_att_name] == next_node]
                    self.generate_id3_tree(False, next_root, next_node, subset, curr_depth)

    def set_depth(self, new_depth):
        self.max_depth = new_depth

    def fit(self, X, y):
        """ 
        Trains model to a given training data set
        """
        data_set = pd.concat([X, y], axis=1)
        data_set_copy = data_set.copy()
        tree = {}
        self.generate_id3_tree(True, tree, None, data_set_copy, 0)
        self.tree = tree

    def predict(self, X):
        """
        A method that predicts classes for given data set
        """
        predictions = []
        for _, row in X.iterrows():
            prediction = self.predict_row(self.tree, row)
            predictions.append(prediction)

        return np.array(predictions)

    def evaluate(self, X, y):
        pred = self.predict(X)
        pred_ok = np.count_nonzero(pred == y)
        pred_all = pred.shape[0]
        acc = pred_ok/pred_all

        return acc
