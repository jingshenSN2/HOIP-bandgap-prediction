import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor as mlp
from preprocessing import raw_preprocessing as pre


def DT():
    X_train, X_test, y_train, y_test, predict_X, features = pre('/Users/wangyizhou/Desktop/机器学习/大作业/data')
