import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import plotly.graph_objects as go

# from sklearn.metrics import confusion_matrix


def visualise_report(report: dict, title: str) -> None:
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
    layout = go.Layout(title=title, title_x=0.5)
    fig = go.Figure(data=[table], layout=layout)
    fig.show()


x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")["HeartDisease"]
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")["HeartDisease"]

xgb = XGBClassifier(
    booster="gbtree",
    colsample_bytree=0.7,
    max_depth=30,
    scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
)
xgb.fit(x_train, y_train)
print(f"XGBoost train acc: {xgb.score(x_train, y_train)}")
print(f"XGBoost test  acc: {xgb.score(x_test, y_test)}")
xgb_pred = xgb.predict(x_test)
# tn, fp, fn, tp = confusion_matrix(y_test, xgb_pred).ravel()
# print(tn, fp, fn, tp)
xgb_report = classification_report(y_test, xgb_pred, output_dict=True)
visualise_report(xgb_report, "The Classification Report of XGBoost")

rf = RandomForestClassifier(class_weight="balanced")
rf.fit(x_train, y_train)
print(f"Random forest train acc: {rf.score(x_train, y_train)}")
print(f"Random forest test  acc: {rf.score(x_test, y_test)}")
rf_pred = rf.predict(x_test)
# tn, fp, fn, tp = confusion_matrix(y_test, rf_pred).ravel()
# print(tn, fp, fn, tp)
rf_report = classification_report(y_test, rf_pred, output_dict=True)
visualise_report(rf_report, "The Classification Report of Random Forest")

dt = DecisionTreeClassifier(max_features="sqrt", class_weight="balanced")
dt.fit(x_train, y_train)
print(f"Decision tree train acc: {dt.score(x_train, y_train)}")
print(f"Decision tree test  acc: {dt.score(x_test, y_test)}")
dt_pred = dt.predict(x_test)
# tn, fp, fn, tp = confusion_matrix(y_test, dt_pred).ravel()
# print(tn, fp, fn, tp)
dt_report = classification_report(y_test, dt_pred, output_dict=True)
visualise_report(dt_report, "The Classification Report of Decision Tree")
