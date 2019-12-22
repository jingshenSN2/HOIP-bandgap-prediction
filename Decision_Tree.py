import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor as dt
from preprocessing import drop_preprocessing as pre
import graphviz
import joblib

from sklearn.model_selection import GridSearchCV as gs


def DecisionTree(data_directory, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre(data_directory, features)
    model = dt(random_state=1)
    grid = gs(estimator=model, param_grid={'criterion': ['mse', 'friedman_mse', 'mae'], 'splitter': ['best', 'random'],
                                           'max_features': ['auto', 'sqrt', 'log2']}, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_.score(X_test, y_test))
    joblib.dump(grid.best_estimator_, 'Best_DecisionTree.dump')
