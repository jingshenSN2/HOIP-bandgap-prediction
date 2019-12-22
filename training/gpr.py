from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import preprocessing as pre
import numpy as np
import pandas as pd
import os

def gpr(data_dir, model_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, features)
    os.chdir(model_dir)
    gpr = GPR(random_state=1)
    grid = GridSearchCV(estimator=gpr, param_grid={'alpha': np.exp(np.log(10)*range(-12, -5)), 'n_restarts_optimizer': [0, 1, 2, 3, 4, 5]})
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_estimator_.score(X_test, y_test))

    joblib.dump(grid.best_estimator_, 'gpr_%d_%.4f.m'%(len(features),grid.best_estimator_.score(X_test, y_test)))


