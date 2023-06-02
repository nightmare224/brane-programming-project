#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load


def split_data_train_test(
    df: pd.DataFrame, label_name: str, test_ratio: float = 0.25
):
    # decimal_feature_names = []
    # for feature_name in list(df.dtypes[df.dtypes == "float64"].index):
    #     if not np.all(df[feature_name].astype(int) == df[feature_name]):
    #         decimal_feature_names.append(feature_name)
    # df = balance_dataset(df, label_name, decimal_feature_names)
    x = df.loc[:, df.columns != label_name]
    y = df.loc[:, label_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)

    return x_train, x_test, y_train, y_test

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




def train_dt(x_train: pd.DataFrame, y_train: pd.DataFrame):
    dt = DecisionTreeClassifier(max_features="sqrt", class_weight="balanced")
    dt.fit(x_train, y_train)
    dt.train_acc = dt.score(x_train, y_train)

    return dt

def train_rf(x_train: pd.DataFrame, y_train: pd.DataFrame):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    rf = RandomForestClassifier(class_weight="balanced")
    rf.fit(x_train, y_train)
    rf.train_acc = rf.score(x_train, y_train)

    return rf

def train_xgb(x_train: pd.DataFrame, y_train: pd.DataFrame):
    xgb = XGBClassifier(
        booster="gbtree",
        colsample_bytree=0.7,
        max_depth=30,
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1]
    )
    xgb.fit(x_train, y_train)
    xgb.train_acc = xgb.score(x_train, y_train)

    return xgb


def test_dt(model, x_test: pd.DataFrame, y_test: pd.DataFrame):
    model.test_acc = model.score(x_test, y_test)

def test_rf(model, x_test: pd.DataFrame, y_test: pd.DataFrame):
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    model.test_acc = model.score(x_test, y_test)

def test_xgb(model, x_test: pd.DataFrame, y_test: pd.DataFrame):
    model.test_acc = model.score(x_test, y_test)

def generate_model(df: pd.DataFrame, label_name: str):
    def model_dump(model, model_name, feature_names):
        model.model_name = model_name
        model.feature_names = feature_names
        dump(model, f"/result/model_{model_name}.joblib")

    x_train, x_test, y_train, y_test = split_data_train_test(df, label_name)
    feature_names = list(x_train.columns)
    
    # train and test model
    dt = train_dt(x_train, y_train)
    test_dt(dt, x_test, y_test)
    model_dump(dt, "decisiontree", feature_names)
    del dt

    rf = train_rf(x_train, y_train)
    test_rf(rf, x_test, y_test)
    model_dump(rf, "randomforest", feature_names)
    del rf

    xgb = train_xgb(x_train, y_train)
    test_xgb(xgb, x_test, y_test)
    model_dump(xgb, "xgboost", feature_names)
    del xgb

    # store test result and dump
    # model = {
    #     "decisiontree": dt,
    #     "randomforest":rf,
    #     "xgboost": xgb
    # }
    # for model_name in model.keys():
    #     # store model metadata
    #     model[model_name].model_name = model_name
    #     model[model_name].feature_names = list(x_train.columns)
    #     # dump model
    #     dump(model[model_name], f"./model_{model_name}.joblib")


if __name__ == "__main__":
    functions = {"generate_model": generate_model}
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # run function
    cmd = sys.argv[1]
    if cmd == "generate_model":
        # load parameter
        filepath = json.loads(os.environ["FILEPATH"])
        label_name = json.loads(os.environ["LABEL_NAME"])

        df = pd.read_csv(f"{filepath}/data_cleaned.csv")
        functions[cmd](df, label_name)


