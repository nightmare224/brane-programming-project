import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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


def ROC_curve(
    model_balanced: any,
    x_test_balanced: pd.DataFrame,
    y_test_balanced: pd.Series,
    model_imbalanced: any,
    x_test_imbalanced: pd.DataFrame,
    y_test_imbalanced: pd.Series,
    model_name: str,
) -> None:
    y_pred_balanced = model_balanced.predict_proba(x_test_balanced)[:, 1]
    fpr_balanced, tpr_balanced, _ = roc_curve(y_test_balanced, y_pred_balanced)
    roc_auc_balanced = roc_auc_score(y_test_balanced, y_pred_balanced)
    y_pred_imbalanced = model_imbalanced.predict_proba(x_test_imbalanced)[:, 1]
    fpr_imbalanced, tpr_imbalanced, _ = roc_curve(y_test_imbalanced, y_pred_imbalanced)
    roc_auc_imbalanced = roc_auc_score(y_test_imbalanced, y_pred_imbalanced)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_balanced,
            y=tpr_balanced,
            mode="lines",
            name="Balanced dataset's ROC curve (area = %0.3f)" % roc_auc_balanced,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr_imbalanced,
            y=tpr_imbalanced,
            mode="lines",
            name="imbalanced dataset's ROC curve (area = %0.3f)" % roc_auc_imbalanced,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
    )
    fig.update_layout(
        title=f"Receiver Operating Characteristic of {model_name}",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        showlegend=True,
    )
    fig.show()


x_train_balanced = pd.read_csv("../data/x_train_balanced.csv")
y_train_balanced = pd.read_csv("../data/y_train_balanced.csv")["HeartDisease"]
x_test_balanced = pd.read_csv("../data/x_test_balanced.csv")
y_test_balanced = pd.read_csv("../data/y_test_balanced.csv")["HeartDisease"]

x_train = pd.read_csv("../data/x_train.csv")
y_train = pd.read_csv("../data/y_train.csv")["HeartDisease"]
x_test = pd.read_csv("../data/x_test.csv")
y_test = pd.read_csv("../data/y_test.csv")["HeartDisease"]


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

xgb_balanced = XGBClassifier(
    booster="gbtree",
    min_child_weight=3,
    max_depth=23,
    learning_rate=0.2,
    gamma=0.6,
    colsample_bytree=0.6,
    n_estimators=80,
    scale_pos_weight=y_train_balanced.value_counts()[0]
    / y_train_balanced.value_counts()[1],
)
xgb_balanced.fit(x_train_balanced, y_train_balanced)
print(
    f"XGBoost_balanced train acc: {xgb_balanced.score(x_train_balanced, y_train_balanced)}"
)
print(
    f"XGBoost_balanced test  acc: {xgb_balanced.score(x_test_balanced, y_test_balanced)}"
)
xgb_pred_balanced = xgb_balanced.predict(x_test_balanced)
# tn, fp, fn, tp = confusion_matrix(y_test_balanced, xgb_pred_balanced).ravel()
# print(tn, fp, fn, tp)
xgb_report_balanced = classification_report(
    y_test_balanced, xgb_pred_balanced, output_dict=True
)
visualise_report(xgb_report_balanced, "The Classification Report of XGBoost_balanced")

ROC_curve(
    xgb_balanced, x_test_balanced, y_test_balanced, xgb, x_test, y_test, "XGBoost"
)


rf = RandomForestClassifier(
    n_estimators=90,
    min_samples_split=3,
    max_depth=20,
    class_weight="balanced",
    criterion="gini",
)
rf.fit(x_train, y_train)
print(f"Random forest train acc: {rf.score(x_train, y_train)}")
print(f"Random forest test  acc: {rf.score(x_test, y_test)}")
rf_pred = rf.predict(x_test)
# tn, fp, fn, tp = confusion_matrix(y_test, rf_pred).ravel()
# print(tn, fp, fn, tp)
rf_report = classification_report(y_test, rf_pred, output_dict=True)
visualise_report(rf_report, "The Classification Report of Random Forest")

rf_balanced = RandomForestClassifier(
    n_estimators=90,
    min_samples_split=9,
    max_depth=22,
    criterion="entropy",
    class_weight="balanced",
)
rf_balanced.fit(x_train_balanced, y_train_balanced)
print(
    f"Random forest_balanced train acc: {rf_balanced.score(x_train_balanced, y_train_balanced)}"
)
print(
    f"Random forest_balanced test  acc: {rf_balanced.score(x_test_balanced, y_test_balanced)}"
)
rf_pred_balanced = rf_balanced.predict(x_test_balanced)
# tn, fp, fn, tp = confusion_matrix(y_test_balanced, rf_pred_balanced).ravel()
# print(tn, fp, fn, tp)
rf_report_balanced = classification_report(
    y_test_balanced, rf_pred_balanced, output_dict=True
)
visualise_report(
    rf_report_balanced, "The Classification Report of Random Forest_balanced"
)

ROC_curve(
    rf_balanced, x_test_balanced, y_test_balanced, rf, x_test, y_test, "Random Forest"
)


dt = DecisionTreeClassifier(max_features="sqrt", class_weight="balanced")
dt.fit(x_train, y_train)
print(f"Decision tree train acc: {dt.score(x_train, y_train)}")
print(f"Decision tree test  acc: {dt.score(x_test, y_test)}")
dt_pred = dt.predict(x_test)
# tn, fp, fn, tp = confusion_matrix(y_test, dt_pred).ravel()
# print(tn, fp, fn, tp)
dt_report = classification_report(y_test, dt_pred, output_dict=True)
visualise_report(dt_report, "The Classification Report of Decision Tree")

dt_balanced = DecisionTreeClassifier(
    min_samples_leaf=7, max_features="sqrt", class_weight="balanced"
)
dt_balanced.fit(x_train_balanced, y_train_balanced)
print(
    f"Decision tree_balanced train acc: {dt_balanced.score(x_train_balanced, y_train_balanced)}"
)
print(
    f"Decision tree_balanced test  acc: {dt_balanced.score(x_test_balanced, y_test_balanced)}"
)
dt_pred_balanced = dt_balanced.predict(x_test_balanced)
# tn, fp, fn, tp = confusion_matrix(y_test_balanced, dt_pred_balanced).ravel()
# print(tn, fp, fn, tp)
dt_report_balanced = classification_report(
    y_test_balanced, dt_pred_balanced, output_dict=True
)
visualise_report(dt_report, "The Classification Report of Decision Tree_balanced")

ROC_curve(
    dt_balanced, x_test_balanced, y_test_balanced, dt, x_test, y_test, "Decision Tree"
)
