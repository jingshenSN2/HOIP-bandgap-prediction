import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.model_selection import GridSearchCV as gs
from preprocessing import drop_preprocessing as pre
import joblib


def MLP(data_directory, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre(data_directory, features)
    model = mlp(random_state=1, max_iter=1000)
    grid = gs(estimator=model,
              param_grid={'hidden_layer_sizes': [(500, 500)], 'activation':
                  ['logistic', 'tanh', 'relu'], 'alpha': np.exp(2.303 * np.arange(-8, 0)),
                          'learning_rate': ['constant']}, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_.score(X_test, y_test))
    joblib.dump(grid.best_estimator_, 'Best_MLP.dump')
