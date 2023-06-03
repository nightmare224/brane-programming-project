#!/usr/bin/env python3

import os
import sys
import json
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Any
from joblib import load
from glob import glob
from jinja2 import FileSystemLoader,Environment


def feature_importance(model: Any) -> None:
    feature_names = model.feature_names
    model_name = model.model_name
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
    fig.write_html(f"/result/feature_importance_{model_name}.html")


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


def model_report(model: Any):
    report = model.report
    model_name = model.model_name
    df = pd.DataFrame(report).transpose()
    df = df.round(2).applymap("{:.2f}".format)
    df.at["accuracy", "precision"] = ""
    df.at["accuracy", "recall"] = ""
    df.at["accuracy", "support"] = df.at["macro avg", "support"]
    df["support"] = df["support"].astype(str).str.split(".").str[0].astype(int)
    df = pd.concat(
        [
            df.iloc[:2],
            pd.DataFrame([[""] * len(df.columns)], index=[""], columns=df.columns),
            df.iloc[2:],
        ]
    )

    table = go.Table(
        header=dict(values=[""] + df.columns.tolist()),
        cells=dict(values=[df.index.tolist()] + [df[col] for col in df.columns]),
    )
    layout = go.Layout(title=f"The Classification Report of {model_name}", title_x=0.5)
    fig = go.Figure(data=[table], layout=layout)
    fig.write_html(f"/result/model_report_{model_name}.html")

def generate_report(filepath):
    imp_filepaths = glob(f"{filepath}/feature_importance*.html")
    ratio_filepaths = glob(f"{filepath}/ratio*.html")
    model_filepaths = glob(f"{filepath}/model_report*.html")

    # feature importance
    feature_imp_item = []
    for filepath in imp_filepaths:
        filename = os.path.basename(filepath)
        model_name = re.match(r"feature_importance_(.*).html", filename).group(1)
        feature_imp_item.append(f"<a class=\"fig\" name=\"{filename}\" href=\"#\">{model_name}</a>")

    # ratio img
    feature_item = []
    for filepath in ratio_filepaths:
        filename = os.path.basename(filepath)
        feature_name = re.match(r"ratio_(.*).html", filename).group(1)
        feature_item.append(f"<a class=\"fig\" name=\"{filename}\" href=\"#\">{feature_name}</a>")

    # model table
    model_item = []
    for filepath in model_filepaths:
        filename = os.path.basename(filepath)
        model_name = re.match(r"model_report_(.*).html", filename).group(1)
        model_item.append(f"<a class=\"fig\" name=\"{filename}\" href=\"#\">{model_name}</a>")


    # render page
    env = Environment(loader=FileSystemLoader('./'))
    template = env.get_template('base.html')
    out = template.render(
        features = feature_item,
        feature_imps = feature_imp_item,
        models = model_item
    )

    # output report
    with open("/result/report.html", 'w+', encoding='utf-8') as f:
        f.write(out)

# The entrypoint of the script
if __name__ == "__main__":
    functions = {
        "feature_importance": feature_importance,
        "ratio_histogram": ratio_histogram,
        "model_report": model_report,
        "generate_report": generate_report
    }
    if len(sys.argv) != 2 or (sys.argv[1] not in functions.keys()):
        print(f"Usage: {sys.argv[0]} {list(functions.keys())}")
        exit(1)

    # run function
    cmd = sys.argv[1]
    if cmd == "feature_importance":
        filepath = json.loads(os.environ["FILEPATH"])
        models = glob(f"{filepath}/model_*.joblib")
        for model_name in models:
            model = load(model_name)
            functions[cmd](model)
            del model
    elif cmd == "ratio_histogram":
        # load parameter
        filepath = json.loads(os.environ["FILEPATH"])
        feature_name = json.loads(os.environ["FEATURE_NAME"])
        label_name = json.loads(os.environ["LABEL_NAME"])
        positive_value = json.loads(os.environ["POSITIVE_VALUE"])
        feature_order = eval(json.loads(os.environ["FEATURE_ORDER"]))

        df = pd.read_csv(f"{filepath}/analysis.csv")
        functions[cmd](
            df, feature_name, label_name, positive_value, feature_order
        )
    elif cmd == "model_report":
        filepath = json.loads(os.environ["FILEPATH"])
        models = glob(f"{filepath}/model_*.joblib")
        for model_name in models:
            model = load(model_name)
            functions[cmd](model)
            del model
    elif cmd == "generate_report":
        filepath = json.loads(os.environ["FILEPATH"])
        functions[cmd](filepath)
