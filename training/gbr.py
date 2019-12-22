from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import preprocessing as pre
import pandas as pd
import os


def gbr(data_dir, model_dir, features):
    X_train, X_test, y_train, y_test, predict_X, features = pre.drop_preprocessing(data_dir, features)
    os.chdir(model_dir)
    gbr = GBR(subsample=1, random_state=1)
    grid = GridSearchCV(estimator=gbr, param_grid={'loss':['ls','lad','huber','quantile'],'n_estimators':range(50,311,20)}, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_.score(X_test, y_test))

    joblib.dump(grid.best_estimator_, 'gbr_%d_%.4f.m'%(len(features),grid.best_estimator_.score(X_test, y_test)))

    df = pd.DataFrame(columns=['pbe_bandgap', 'ml_bandgap'])
    df['pbe_bandgap'] = y_test
    df['ml_bandgap'] = grid.best_estimator_.predict(X_test)
    print(df)

