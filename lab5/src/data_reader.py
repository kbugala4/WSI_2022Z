from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class DataReader(object):
    def __init__(self, data, labels):
        self.data = data / data.max()
        self.labels = np.array(pd.get_dummies(labels))

    def split_data(self):
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            self.data, self.labels, test_size=0.4, random_state=42
        )
        test_data, valid_data, test_labels, valid_labels = train_test_split(
            temp_data, temp_labels, test_size=0.5, random_state=42
        )

        return (
            train_data,
            train_labels,
            test_data,
            test_labels,
            valid_data,
            valid_labels,
        )
