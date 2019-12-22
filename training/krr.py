from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import preprocessing as pre
import numpy as np
import pandas as pd
import os

def krr(data_dir, model_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, features)
    os.chdir(model_dir)
    krr = KernelRidge()
    alphas = np.arange(0.1,0.25,0.02)#[pow(10,x) for x in np.arange(-1.5,0,0.02)]
    gammas = np.arange(5,8,0.2)#[pow(10,x) for x in np.arange(0,1,0.02)]
    grid = GridSearchCV(estimator=krr, param_grid={'alpha': alphas, 'kernel':['linear','rbf','sigmoid', 'poly'], 'gamma': gammas, 'degree' : [2,3,4,5]}, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_.score(X_test, y_test))

    joblib.dump(grid.best_estimator_, 'krr_%d_%.4f.m'%(len(features),grid.best_estimator_.score(X_test, y_test)))
