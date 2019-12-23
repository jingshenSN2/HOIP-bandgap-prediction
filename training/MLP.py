import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.model_selection import GridSearchCV as gs
from preprocessing import drop_preprocessing as pre
import joblib
import os

def MLP(data_directory, model_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre(data_directory, features)
    os.chdir(model_dir)
    model = mlp(random_state=1, max_iter=10000)
    grid = gs(estimator=model,
              param_grid={'hidden_layer_sizes': [(500, 500)], 'activation':
                  ['logistic', 'tanh', 'relu'], 'alpha': np.exp(2.303 * np.arange(-8, 0)),
                          'learning_rate': ['constant']}, cv=5, n_jobs=6)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_.score(X_test, y_test))

    joblib.dump(grid.best_estimator_, 'mlp_%d_%.4f.m'%(len(features),grid.best_estimator_.score(X_test, y_test)))

    df = pd.DataFrame(columns=['ml_bandgap', 'pbe_bandgap'])
    df['pbe_bandgap'] = y_test
    df['ml_bandgap'] = grid.best_estimator_.predict(X_test)
    print(df)