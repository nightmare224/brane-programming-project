from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def label_encoding(df: DataFrame, column_names: list[str]) -> DataFrame:
    for column in column_names:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

def ordinal_encoding(df: DataFrame, column_names: list[str], column_categories: list[list]) -> DataFrame:
    df[column_names] = OrdinalEncoder(categories=column_categories).fit_transform(df[column_names])
    return df

def one_hot_encoding(df: DataFrame, column_names: list[str]) -> DataFrame:
    for column in column_names:
        result = OneHotEncoder(sparse_output=False).fit_transform(df[column].to_numpy().reshape(-1, 1)).transpose()
        base_index = df.columns.get_loc(column)
        df.drop(columns=column, inplace=True)
        for i in range(len(result)):
            df.insert(base_index + i, f"{column}_{i}", result[i])
    return df

df = read_csv("../data/raw.csv")
# df = label_encoding(df, ["HeartDisease", "Smoking"])
# df = one_hot_encoding(df, ["HeartDisease", "Smoking"])
# df = ordinal_encoding(df, ["HeartDisease", "Smoking"], [["No", "Yes"], ["No", "Yes"]])
# print(df)