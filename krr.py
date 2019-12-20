from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import preprocessing as pre
import numpy as np
import pandas as pd

def krr(data_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, features)
    krr = KernelRidge()
    grid = GridSearchCV(estimator=krr, param_grid={'alpha': np.logspace(-9,-1,9), 'kernel':['linear','rbf','sigmoid', 'poly'], 'gamma': np.logspace(-5, 5, 11)})
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_estimator_.score(X_test, y_test))
    df = pd.DataFrame(columns=['pbe_bandgap','ml_bandgap'])
    df['pbe_bandgap'] = y_test
    df['ml_bandgap'] = grid.best_estimator_.predict(X_test)
    print(df)