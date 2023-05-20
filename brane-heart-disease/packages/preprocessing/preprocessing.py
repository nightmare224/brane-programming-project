#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd

def one_hot_encoding(column_name: str, df: pd.DataFrame) -> pd.DataFrame:
    return df

def ordinal_encoding(column_name: str, df: pd.DataFrame) -> pd.DataFrame:
    return df

# The entrypoint of the script
if __name__ == "__main__":
    functions = {"one_hot_encoding": one_hot_encoding, "ordinal_encoding": ordinal_encoding}
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # load args from environment variable
    column_name = json.loads(os.environ["COLUMN_NAME"])
    file_name = json.loads(os.environ["FILE_NAME"])
    df = pd.read_csv(file_name)

    cmd = sys.argv[0]
    if cmd == "one_hot_encoding":
        result = functions[cmd](column_name, df)
        pd.to_csv("/result/result.csv", index = False)
    elif cmd == "ordinal_encoding":
        result = functions[cmd](column_name, df)
        pd.to_csv("/result/result.csv", index = False)