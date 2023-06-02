import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

x_train = pd.read_csv("../data/x_train.csv")
y_train = pd.read_csv("../data/y_train.csv")
x_test = pd.read_csv("../data/x_test.csv")
y_test = pd.read_csv("../data/y_test.csv")

xgb = XGBClassifier(
    booster="gbtree",
    colsample_bytree=0.7,
    gamma=0.2,
    learning_rate=0.5,
    max_depth=30,
    min_child_weight=3,
    n_estimators=150,
)
xgb.fit(x_train, y_train)
print(f"XGBoost train acc: {xgb.score(x_train, y_train)}")
print(f"XGBoost test  acc: {xgb.score(x_test, y_test)}")

rf = RandomForestClassifier()
rf.fit(x_train, y_train.to_numpy().flatten())
print(f"Random forest train acc: {rf.score(x_train, y_train)}")
print(f"Random forest test  acc: {rf.score(x_test, y_test)}")

dt = DecisionTreeClassifier(max_features="sqrt")
dt.fit(x_train, y_train)
print(f"Decision tree train acc: {dt.score(x_train, y_train)}")
print(f"Decision tree test  acc: {dt.score(x_test, y_test)}")
