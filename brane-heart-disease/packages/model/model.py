#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load


def generate_data_train_test(
    df: pd.DataFrame, label_name: str, test_ratio: float = 0.25
):
    df = balance_dataset(df, label_name, list(df.dtypes[df.dtypes == "float64"].index))
    x = df.loc[:, df.columns != label_name]
    y = df.loc[:, label_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)

    x_train.to_csv("/result/x_train.csv")
    x_test.to_csv("/result/x_test.csv")
    y_train.to_csv("/result/y_train.csv")
    y_test.to_csv("/result/y_test.csv")


def balance_dataset(
    df: pd.DataFrame,
    label_name: str,
    decimal_allowed_feature_names: List[str],
    ratio: float = 1.0,
) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    features = df.loc[:, df.columns != label_name]
    label = df.loc[:, label_name]
    sm = SMOTE(sampling_strategy=ratio)
    new_features, new_label = sm.fit_resample(features, label)
    rounding_column_names = new_features.columns.drop(decimal_allowed_feature_names)
    new_features[rounding_column_names] = new_features[rounding_column_names].round()
    return pd.concat([new_label, new_features], axis=1)


def train(X_train: pd.DataFrame, Y_train: pd.DataFrame):
    feature_names = list(X_train.columns)
    x_train = np.array(X_train)
    y_train = np.array(Y_train[Y_train.columns[0]])
    rf = RandomForestClassifier(random_state=0)
    rf.fit(x_train, y_train)
    rf.feature_names = feature_names
    dump(rf, "/result/model_rf.joblib")


def test(model, X_test, Y_test):
    pass


def predict(model, X_test):
    pass


if __name__ == "__main__":
    functions = {"train": train, "generate_data_train_test": generate_data_train_test}
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # run function
    cmd = sys.argv[1]
    if cmd == "train":
        # load parameter
        filepath_x = json.loads(os.environ["FILEPATH_X"])
        filepath_y = json.loads(os.environ["FILEPATH_Y"])

        df_x = pd.read_csv(filepath_x)
        df_y = pd.read_csv(filepath_y)
        functions[cmd](df_x, df_y)
    elif cmd == "generate_data_train_test":
        # load parameter
        filepath = json.loads(os.environ["FILEPATH"])
        label_name = json.loads(os.environ["LABEL_NAME"])

        df = pd.read_csv(filepath)
        functions[cmd](df, label_name)
