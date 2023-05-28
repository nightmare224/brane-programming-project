import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

def train(X_train, Y_train):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train[Y_train.columns[0]])
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, Y_train)
    dump(rf, 'model_rf.joblib')