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
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from joblib import dump, load


def split_data_train_test(
    df: pd.DataFrame, label_name: str, test_ratio: float = 0.25
):
    # train test split
    x = df.loc[:, df.columns != label_name]
    y = df.loc[:, label_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)

    # balance train data
    train = pd.concat([y_train, x_train], axis=1)
    decimal_feature_names = []
    for feature_name in list(df.dtypes[df.dtypes == "float64"].index):
        if not np.all(df[feature_name].astype(int) == df[feature_name]):
            decimal_feature_names.append(feature_name)
    train = balance_dataset(train, label_name, decimal_feature_names)
    y_train = train[label_name]
    x_train = train.drop(label_name, axis=1)

    return x_train, x_test, y_train, y_test

def balance_dataset(
    df: pd.DataFrame,
    label_name: str,
    decimal_allowed_feature_names: List[str],
    ratio: float = 0.5,
) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    features = df.loc[:, df.columns != label_name]
    label = df.loc[:, label_name]
    sm = RandomOverSampler(sampling_strategy=ratio)
    new_features, new_label = sm.fit_resample(features, label)
    rounding_column_names = new_features.columns.drop(decimal_allowed_feature_names)
    new_features[rounding_column_names] = new_features[rounding_column_names].round()
    return pd.concat([new_label, new_features], axis=1)


def train_dt(x_train: pd.DataFrame, y_train: pd.DataFrame):
    dt = DecisionTreeClassifier(min_samples_leaf=7, max_features="sqrt", class_weight="balanced")
    dt.fit(x_train, y_train)
    dt.train_acc = dt.score(x_train, y_train)

    return dt

def train_rf(x_train: pd.DataFrame, y_train: pd.DataFrame):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    rf = RandomForestClassifier(
        n_estimators=90,
        min_samples_split=9,
        max_depth=22,
        criterion="entropy",
        class_weight="balanced",
    )
    rf.fit(x_train, y_train)
    rf.train_acc = rf.score(x_train, y_train)

    return rf

def train_xgb(x_train: pd.DataFrame, y_train: pd.DataFrame):
    xgb = XGBClassifier(
        booster="gbtree",
        min_child_weight=3,
        max_depth=23,
        learning_rate=0.2,
        gamma=0.6,
        colsample_bytree=0.6,
        n_estimators=80,
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
    )
    xgb.fit(x_train, y_train)
    xgb.train_acc = xgb.score(x_train, y_train)

    return xgb


def test_dt(model, x_test: pd.DataFrame, y_test: pd.DataFrame):
    # model.test_acc = model.score(x_test, y_test)
    pred = model.predict(x_test)
    report = classification_report(y_test, pred, output_dict=True)
    model.report = report

def test_rf(model, x_test: pd.DataFrame, y_test: pd.DataFrame):
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # model.test_acc = model.score(x_test, y_test)
    pred = model.predict(x_test)
    report = classification_report(y_test, pred, output_dict=True)
    model.report = report

def test_xgb(model, x_test: pd.DataFrame, y_test: pd.DataFrame):
    # model.test_acc = model.score(x_test, y_test)
    pred = model.predict(x_test)
    report = classification_report(y_test, pred, output_dict=True)
    model.report = report

def generate_model(df: pd.DataFrame, label_name: str):
    def model_dump(model, model_name, feature_names):
        model.model_name = model_name
        model.feature_names = feature_names
        dump(model, f"/result/model_{model_name}.joblib", compress = 3)

    # generate train and test data
    category0_count, category1_count = df[label_name].value_counts()
    category_ratio = 0.5
    test_ratio = 0.25
    x_train, x_test, y_train, y_test = split_data_train_test(
        df,
        label_name,
        (1 + category_ratio) * test_ratio / (test_ratio + category_ratio * test_ratio + 1 + category1_count / category0_count),
    )

    # train and test model
    feature_names = list(x_train.columns)    
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


