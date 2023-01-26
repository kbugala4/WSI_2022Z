import pandas as pd


class DataReader():
    def read_df(self, file):
        df = pd.read_csv(file, delimiter=";")
        df = df.drop('id', axis=1)
        return df