import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE


def label_encoding(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    for column_name in column_names:
        enc = LabelEncoder()
        df[column_name] = enc.fit_transform(df[column_name])
    return df


def ordinal_encoding(
    df: pd.DataFrame, column_names: list[str], columns_categories: list[list[str]]
) -> pd.DataFrame:
    enc = OrdinalEncoder(categories=columns_categories)
    df[column_names] = enc.fit_transform(df[column_names])
    return df


def one_hot_encoding(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    for column_name in column_names:
        enc = OneHotEncoder(sparse_output=False)
        new_columns = enc.fit_transform(
            df[column_name].to_numpy().reshape(-1, 1)
        ).transpose()
        base_index = df.columns.get_loc(column_name)
        df.drop(columns=column_name, inplace=True)
        for i, (new_column_name, new_column) in enumerate(
            zip(enc.get_feature_names_out([column_name]), new_columns)
        ):
            df.insert(base_index + i, new_column_name, new_column)
    return df


def balance_dataset(
    df: pd.DataFrame, label_name: str, ratio: float = 1.0
) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    features = df.loc[:, df.columns != label_name]
    label = df.loc[:, label_name]
    sm = SMOTE(sampling_strategy=ratio)
    try:
        new_features, new_label = sm.fit_resample(features, label)
    except ValueError:
        raise ValueError("Dataset contains non-numeric value")
    else:
        return pd.concat([new_label, new_features], axis=1)


def split_significance(
    df: pd.DataFrame, label: pd.DataFrame, is_sig: bool, alpha: float = 0.05
) -> pd.DataFrame:
    result = pd.DataFrame()
    label = label.squeeze()
    if is_sig:
        for name, values in df.items():
            _, pvalue, _, _ = chi2_contingency(pd.crosstab(values, label))
            if pvalue < alpha:
                result[name] = values
    else:
        for name, values in df.items():
            _, pvalue, _, _ = chi2_contingency(pd.crosstab(values, label))
            if pvalue >= alpha:
                result[name] = values
    return result


df = pd.read_csv("../data/raw.csv")
label_name = "HeartDisease"
# df = label_encoding(
#     df,
#     [
#         "HeartDisease",
#         "Smoking",
#         "AlcoholDrinking",
#         "Stroke",
#         "DiffWalking",
#         "Sex",
#         "AgeCategory",
#         "Race",
#         "Diabetic",
#         "PhysicalActivity",
#         "GenHealth",
#         "Asthma",
#         "KidneyDisease",
#         "SkinCancer",
#     ],
# )
# df = label_encoding(df, ["HeartDisease", "Smoking"])
# df = one_hot_encoding(df, ["HeartDisease", "Smoking"])
# df = ordinal_encoding(df, ["HeartDisease", "Smoking"], [["No", "Yes"], ["No", "Yes"]])
# df = balance_dataset(df, label_name)
# df = split_significance(df, pd.DataFrame(df[label_name]), True)
print(df)
# x = df.loc[:, df.columns != label_name]
# y = df.loc[:, label_name]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# x_train.to_csv("x_train.csv")
# x_test.to_csv("x_test.csv")
# y_train.to_csv("y_train.csv")
# y_test.to_csv("y_test.csv")
