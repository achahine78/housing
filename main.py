import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def load_data(filename):
    #loads data from csv into pandas dataframe
    return pd.read_csv(filename)

def generate_income_category(housing):
    #generate income categories by dividing by 1.5 and merging all categories greater than 5 into 5
    #to be used in training/test set generation
    housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


def split_training_data(housing):
    #splits our dataset into a test set and training set using sklearn's StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    return strat_train_set, strat_test_set

def remove_income_category(train_set, test_set):
    #return data to original state
    for data_set in (train_set, test_set):
        data_set.drop(["income_cat"], axis=1, inplace=True)

def run():
    housing = load_data("housing.csv")
    generate_income_category(housing)
    train_set, test_set = split_training_data(housing)
    remove_income_category(train_set, test_set)

if __name__ == "__main__":
   run()