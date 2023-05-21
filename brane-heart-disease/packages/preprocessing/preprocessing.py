#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def label_encoding(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    for column in column_names:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

def ordinal_encoding(df: pd.DataFrame, column_names: str, columns_categories: list) -> pd.DataFrame:
    
    return df

def one_hot_encoding(df: pd.DataFrame, column_names: str) -> pd.DataFrame:
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

    # load args from environment variable
    column_names = json.loads(os.environ["COLUMN_NAMES"])
    filepath = json.loads(os.environ["FILEPATH"])
    # the intermidate result seems would pass the directory
    if os.path.isdir(filepath):
        filepath = f"{filepath}/result.csv"
    df = pd.read_csv(filepath)
    
    result_filepath = "/result/result.csv"
    cmd = sys.argv[1]
    if cmd == "label_encoding":
        column_names = eval(column_names)
        result = functions[cmd](df, column_names)
        result.to_csv(result_filepath, index = False)
    elif cmd == "ordinal_encoding":
        result = functions[cmd](df, column_names, [])
        result.to_csv(result_filepath, index = False)
        # from glob import glob
        # filepath = glob(f"{str(filepath)}/*", recursive = True)
        # print(yaml.dump({"result": str(filepath)}))
    elif cmd == "one_hot_encoding":
        result = functions[cmd](df, column_names)
        result.to_csv(result_filepath, index = False)