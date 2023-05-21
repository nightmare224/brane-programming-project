#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd

def label_encoding(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    return df

def ordinal_encoding(df: pd.DataFrame, column_names: str, column_category: list) -> pd.DataFrame:
    return df

def one_hot_encoding(df: pd.DataFrame, column_names: str) -> pd.DataFrame:
    return df

# The entrypoint of the script
if __name__ == "__main__":
    functions = {"one_hot_encoding": one_hot_encoding, "ordinal_encoding": ordinal_encoding}
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # load args from environment variable
    column_name = json.loads(os.environ["COLUMN_NAME"])
    filepath = json.loads(os.environ["FILEPATH"])
    # the intermidate result seems would pass the directory
    if os.path.isdir(filepath):
        filepath = f"{filepath}/result.csv"
    df = pd.read_csv(filepath)
    
    result_filepath = "/result/result.csv"
    cmd = sys.argv[1]
    if cmd == "one_hot_encoding":
        result = functions[cmd](df, column_name)
        result.to_csv(result_filepath, index = False)
    elif cmd == "ordinal_encoding":
        result = functions[cmd](df, column_name, [])
        result.to_csv(result_filepath, index = False)
        # from glob import glob
        # filepath = glob(f"{str(filepath)}/*", recursive = True)
        # print(yaml.dump({"result": str(filepath)}))