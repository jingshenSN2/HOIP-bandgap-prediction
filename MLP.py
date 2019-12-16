import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.model_selection import GridSearchCV as gs
from preprocessing import raw_preprocessing as pre


def MLP():
    X_train, X_test, y_train, y_test, predict_X, features = pre('/Users/wangyizhou/Desktop/机器学习/大作业/data')
    model = mlp()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
