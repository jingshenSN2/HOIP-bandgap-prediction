from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.model_selection import GridSearchCV
import preprocessing as pre
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gpr(data_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, features)
    gpr = GPR(random_state=1)
    grid = GridSearchCV(estimator=gpr, param_grid={'alpha': np.exp(np.log(10)*range(-12, -5)), 'n_restarts_optimizer': [0, 1, 2, 3, 4, 5]})
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_estimator_.score(X_test, y_test))
    df = pd.DataFrame(columns=['pbe_bandgap', 'ml_bandgap'])
    df['pbe_bandgap'] = y_test
    df['ml_bandgap'] = grid.best_estimator_.predict(X_test)
    print(df)
    #plt.scatter(df['ml_bandgap'], df['pbe_bandgap'])
    #plt.show()


