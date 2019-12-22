from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import preprocessing as pre
import numpy as np
import pandas as pd
import os

def svr(data_dir, model_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, features)
    os.chdir(model_dir)
    svr = SVR(max_iter=100000)
    grid = GridSearchCV(estimator=svr, param_grid={'kernel':['linear','rbf','sigmoid', 'poly'], 'gamma': np.logspace(-3, 3, 20), 'C':np.logspace(-3, 3, 20)})
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_estimator_.score(X_test, y_test))

    joblib.dump(grid.best_estimator_, 'svr_%d_%.4f.m'%(len(features),grid.best_estimator_.score(X_test, y_test)))
