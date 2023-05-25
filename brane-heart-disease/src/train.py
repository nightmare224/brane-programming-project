from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dt = DecisionTreeClassifier()

X_train = pd.read_csv("../data/x_train.csv")
Y_train = pd.read_csv("../data/y_train.csv")

X_test = pd.read_csv("../data/x_test.csv")

dt.fit(X_train, Y_train)

y_pred = dt.predict(X_test)