#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
from typing import List, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def label_encoding(df: pd.DataFrame, column_names: Union[List[str], str]) -> pd.DataFrame:
    if type(column_names) is str:
        column_names = [column_names]

    for column in column_names:
        df[column] = LabelEncoder().fit_transform(df[column])
    
    return df

def ordinal_encoding(df: pd.DataFrame, column_names: List[str], columns_categories: List[List[str]]) -> pd.DataFrame:
    df[column_names] = OrdinalEncoder(categories=columns_categories).fit_transform(df[column_names])
    return df

def one_hot_encoding(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    for column in column_names:
        result = OneHotEncoder(sparse_output=False).fit_transform(df[column].to_numpy().reshape(-1, 1)).transpose()
        base_index = df.columns.get_loc(column)
        df.drop(columns=column, inplace=True)
        for i in range(len(result)):
            df.insert(base_index + i, f"{column}_{i}", result[i])
    return df

# The entrypoint of the script
if __name__ == "__main__":
    functions = {
        "label_encoding": label_encoding,
        "ordinal_encoding": ordinal_encoding,
        "one_hot_encoding": one_hot_encoding, 
    }
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # load args from environment variable (common)
    filepath = json.loads(os.environ["FILEPATH"])
    column_names = eval(json.loads(os.environ["COLUMN_NAMES"]))
    # column_names = eval(os.environ["COLUMN_NAMES"])


    # the intermidate result seems would pass the directory
    if os.path.isdir(filepath):
        filepath = f"{filepath}/data_cleaned.csv"
    df = pd.read_csv(filepath)
    
    # run function
    result_filepath = "/result/data_cleaned.csv"
    cmd = sys.argv[1]
    if cmd == "label_encoding":
        result = functions[cmd](df, column_names)
        result.to_csv(result_filepath, index = False)
    elif cmd == "ordinal_encoding":
        columns_categories = eval(json.loads(os.environ["COLUMNS_CATEGORIES"]))
        result = functions[cmd](df, column_names, columns_categories)
        result.to_csv(result_filepath, index = False)
    elif cmd == "one_hot_encoding":
        result = functions[cmd](df, column_names)
        result.to_csv(result_filepath, index = False)