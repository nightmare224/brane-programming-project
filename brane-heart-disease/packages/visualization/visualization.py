#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Any
from joblib import load


def feature_importance(model: Any, feature_names: List) -> None:
    imp = model.feature_importances_
    imp_sorted = pd.Series(imp, index=feature_names).sort_values(ascending=True)
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


def ratio_histogram(
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
    total_ratio = total_cnt / df.shape[0]
    if feature_order:
        positive_ratio = positive_ratio[feature_order]
    feature_category = positive_ratio.index

    fig = go.Figure(data = [
        go.Bar(x=feature_category, y=positive_ratio * 100, name=f"Positive ratio (compared to category number)"),
        go.Bar(x=feature_category, y=total_ratio * 100, name=f"Category Ratio (compared to total number)"),

    ])
    fig.update_layout(barmode='group')
    fig.update_layout(title_text=f"<b>The ratio of {feature_name} in {label_name}</b>")
    fig.update_yaxes(title="Percentage (%)")
    fig.update_xaxes(title=feature_name)
    fig.update_layout(
       xaxis = dict(
          tickmode = 'linear',
          tick0 = 1,
          dtick = 1
       )
    )
    fig.write_html(f"/result/ratio_{feature_name}.html")

# The entrypoint of the script
if __name__ == "__main__":
    functions = {
        "feature_importance": feature_importance,
        "ratio_histogram": ratio_histogram,
    }
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # run function
    cmd = sys.argv[1]
    if cmd == "feature_importance":
        filepath = json.loads(os.environ["FILEPATH"])
        model = load(filepath)
        functions[cmd](model)
    elif cmd == "ratio_histogram":
        # load parameter
        filepath = json.loads(os.environ["FILEPATH"])
        feature_name = json.loads(os.environ["FEATURE_NAME"])
        label_name = json.loads(os.environ["LABEL_NAME"])
        positive_value = json.loads(os.environ["POSITIVE_VALUE"])
        feature_order = eval(json.loads(os.environ["FEATURE_ORDER"]))

        df = pd.read_csv(filepath)
        functions[cmd](
            df, feature_name, label_name, positive_value, feature_order
        )
