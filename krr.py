from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import preprocessing as pre

def krr(features):
    X_train, X_test, y_train, y_test, predict_X, feature_list = pre.drop_preprocessing(features)
    reg = KernelRidge(kernel='linear',degree=3)
    grid = GridSearchCV(estimator=reg, n_jobs=-1)
    alpha_list = []
    gamma_list = []
