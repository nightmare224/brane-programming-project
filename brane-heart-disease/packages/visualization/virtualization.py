import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Any
from plotly.subplots import make_subplots


def feature_importance_rf(model: Any, feature_names: List):
    imp = model.feature_importances_
    imp_sorted = pd.Series(imp, index=feature_names).sort_values(ascending=True)
    # sns.barplot(y=imp_sorted.index, x=imp_sorted)
    # plt.show()
    df = pd.DataFrame(
        {"Feature Name": imp_sorted.index, "Feature Importance": imp_sorted}
    )
    fig = px.bar(
        df,
        x="Feature Importance",
        y="Feature Name",
        color="Feature Importance",
        orientation="h",
    )
    fig.update_layout(title=f"<b>Feature Importance</b>")
    fig.update_coloraxes(showscale=False)
    fig.write_image("feature_importance_rf.png")

def total_histogram(
    df: pd.DataFrame,
    feature_name: str,
    feature_order: List = [],
):
    total_cnt = df.groupby([feature_name])[feature_name].count()
    if feature_order:
        total_cnt = total_cnt[feature_order]
    feature_category = total_cnt.index

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=feature_category,
            y=total_cnt,
            # marker=dict(color=["lightgray"] * 9 + ["darkred"] * 4),
            name=f"The total of {feature_name}",
        ),
        secondary_y=False,
    )
    fig.update_layout(title=f"<b>The total of {feature_name}</b>")
    fig.update_yaxes(title=f"The number of {feature_name}", rangemode="tozero", secondary_y=False)
    fig.update_xaxes(title=feature_name)
    fig.write_image(f"total_{feature_name}.png")

def positive_ratio_histogram(
    df: pd.DataFrame,
    feature_name: str,
    label_name: str,
    positive_value: Any = 1,
    feature_order: List = [],
):
    positive_cnt = (
        df[df[label_name] == positive_value].groupby([feature_name])[label_name].count()
    )
    total_cnt = df.groupby([feature_name])[label_name].count()
    positive_ratio = positive_cnt / total_cnt
    if feature_order:
        positive_ratio = positive_ratio[feature_order]
    feature_category = positive_ratio.index

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=feature_category,
            y=positive_ratio,
            # marker=dict(color=["lightgray"] * 9 + ["darkred"] * 4),
            name=f"The ratio of {feature_name} in {label_name}",
        ),
        secondary_y=False,
    )
    fig.update_layout(title=f"<b>The ratio of {feature_name} in {label_name}</b>")
    fig.update_yaxes(title="Ratio", rangemode="tozero", secondary_y=False)
    fig.update_xaxes(title=feature_name)
    fig.write_image(f"positive_ratio_{feature_name}.png")


# The entrypoint of the script
if __name__ == "__main__":
    functions = {
        "positive_ratio_histogram": positive_ratio_histogram,
    }
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # TODO: make it like in preprocessing.py format and write container.ymal
    # filepath is analysis.csv
    filepath = json.loads(os.environ["FILEPATH"])
    df = pd.read_csv(filepath)

    ### some important feature
    # age
    positive_ratio_histogram(df, "AgeCategory", "HeartDisease", "Yes")
    total_histogram(df, "AgeCategory")

    # bmi
    positive_ratio_histogram(
        df,
        "BMICategory",
        "HeartDisease",
        "Yes",
        ["Underweight", "Normal weight", "Overweight", "Obesity"],
    )
    total_histogram(df, "BMICategory", ["Underweight", "Normal weight", "Overweight", "Obesity"])

    # sleep time
    positive_ratio_histogram(df, "SleepTime", "HeartDisease", "Yes")
    total_histogram(df, "SleepTime")