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
    df: pd.DataFrame,
    label_name: str,
    decimal_allowed_feature_names: list[str],
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


def select_significant_features(
    df: pd.DataFrame, label_name: str, is_sig: bool, alpha: float = 0.05
) -> pd.DataFrame:
    result = pd.DataFrame()
    label = df[label_name]
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


def split_data_train_test(
    df: pd.DataFrame, label_name: str, test_ratio: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df.loc[:, df.columns != label_name]
    y = df.loc[:, label_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)
    return x_train, x_test, y_train, y_test


df = pd.read_csv("../data/raw.csv")
label_name = "HeartDisease"
df = label_encoding(
    df,
    [
        "HeartDisease",
        "Smoking",
        "AlcoholDrinking",
        "Stroke",
        "DiffWalking",
        "Sex",
        "AgeCategory",
        "Race",
        "Diabetic",
        "PhysicalActivity",
        "Asthma",
        "KidneyDisease",
        "SkinCancer",
    ],
)
df = ordinal_encoding(
    df, ["GenHealth"], [["Poor", "Fair", "Good", "Very good", "Excellent"]]
)
# df = one_hot_encoding(df, ["Race"])
# df = select_significant_features(df, label_name, True)
value0_count, value1_count = df[label_name].value_counts()
x_train, x_test, y_train, y_test = split_data_train_test(
    df, label_name, value0_count / (3 * value0_count + 2 * value1_count)
)
train = pd.concat([y_train, x_train], axis=1)
train = balance_dataset(train, label_name, ["BMI"])
y_train = train[label_name]
x_train = train.drop(label_name, axis=1)

x_train.to_csv("../data/x_train_balanced.csv")
x_test.to_csv("../data/x_test_balanced.csv")
y_train.to_csv("../data/y_train_balanced.csv")
y_test.to_csv("../data/y_test_balanced.csv")
