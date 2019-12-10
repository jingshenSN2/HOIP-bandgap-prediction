from sklearn.ensemble import GradientBoostingRegressor as GBR

def gbr(X_train, X_test, y_train, y_test,features):
    reg = GBR(random_state = 1)
    reg.fit(X_train,y_train)
    print(reg.score(X_test,y_test))
    print(reg.feature_importances_)
