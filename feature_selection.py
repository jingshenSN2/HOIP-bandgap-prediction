from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import preprocessing as pre
import matplotlib.pyplot as plt


def __ensemble_test(type, X_train, X_test, y_train, y_test):
    if type.lower() == 'gbr':
        reg = GBR(n_estimators=100, random_state=1)
    elif type.lower() == 'rfr':
        reg = RFR(n_estimators=100, random_state=1)
    elif type.lower() == 'abr':
        reg = ABR(n_estimators=100, random_state=1)
    elif type.lower() == 'etr':
        reg = ETR(n_estimators=100, random_state=1)
    reg.fit(X_train, y_train)
    return reg, reg.score(X_test, y_test), reg.feature_importances_


def __plot(type, df, cwdm):
    plt.scatter(df['number'], df['reg_score'], marker='o', c='black', label='R**2')
    plt.scatter(df['number'], df['adj. r**2'], marker='o', c='blue', label='Adj. R**2')
    plt.xlabel('N_feature')
    plt.ylabel('Score of ' + type + ' model')
    plt.axis([-1, 31, 0, 1])
    plt.legend(loc=8)
    plt.savefig(cwdm + 'feature_score_' + type + '.png')
    plt.close()


def feature_selector_ensemble(type, cwdd, cwdm):
    X_train, X_test, y_train, y_test, predict_X, features = pre.raw_preprocessing(cwdd)
    reg_list = []
    local_feature = features
    df = pd.DataFrame(columns=['number', 'reg_score', 'adj. r**2', 'feature_list'])
    while len(local_feature) >= 1:
        reg, score, weight = __ensemble_test(type, X_train, X_test, y_train, y_test)
        reg_list.append(reg)
        df = df.append(pd.DataFrame([[len(local_feature), score,
                                      1 - (1 - score * score) * (40) / (40 - len(local_feature)),
                                      ', '.join(local_feature)]], columns=df.columns), ignore_index=True)
        low = np.argmin(weight)
        del local_feature[low]
        X_train = np.delete(X_train, low, axis=1)
        X_test = np.delete(X_test, low, axis=1)
    df.to_csv(cwdm + 'feature_selection_%s.csv' % type, index=False)
    __plot(type, df, cwdm)
    return reg_list


def __mlp_test(X_train, X_test, y_train, y_test):
    reg = MLPRegressor(hidden_layer_sizes=(20,), max_iter=10000, random_state=1)
    reg.fit(X_train, y_train)
    return reg, reg.score(X_test, y_test)


def __pca_test(dem, X_tr, X_te, y_train, y_test):
    reg = KernelPCA(kernel='linear', n_components=dem, random_state=1)
    re = reg.fit(np.vstack((X_tr, X_te)))
    X_train = re.transform(X_tr)
    X_test = re.transform(X_te)
    reg, score = __mlp_test(X_train, X_test, y_train, y_test)
    print(score)
    return reg, score


def feature_selector_pca(X_tr, X_te, y_tr, y_te, feature, cwd):
    reg_list = []
    X_train, X_test, y_train, y_test = X_tr, X_te, y_tr, y_te
    local_feature = feature
    df = pd.DataFrame(columns=['number', 'reg_score', 'adj. r**2'])
    for i in range(2, len(local_feature) + 1):
        reg, score = __pca_test(i, X_train, X_test, y_train, y_test)
        df = df.append(pd.DataFrame([[i, score, 1 - (1 - score * score) * (40) / (40 - i)]], columns=df.columns),
                       ignore_index=True)
    df.to_csv(cwd + 'feature_selection_pca.csv', index=False)
    return reg_list
