from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import GridSearchCV
import preprocessing as pre
import numpy as np

def gbr(data_dir, f):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, f)
    gbr = GBR(subsample=1, random_state=1)
    grid = GridSearchCV(estimator=gbr, param_grid={'loss':['ls','lad','huber','quantile'],'n_estimators':range(50,311,20)}, cv=5) #
    grid.fit(X_train,y_train)
    print(grid.best_params_, grid.best_score_)
    print(grid.best_estimator_.score(X_test, y_test))