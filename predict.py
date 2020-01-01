import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


def predict(model_directory, predict_directory):
    data = pd.read_csv('unknown_comb.csv', header=0)
    os.chdir(model_directory)
    gbr_4 = joblib.load('gbr_4_0.9669.m')
    X_pred = data[['P_A', 'P_B', 'X_p-electron', 'VE_B']]
    scaler = MinMaxScaler()
    y_pred = gbr_4.predict(scaler.fit_transform(X_pred))
    print(gbr_4.feature_importances_)
    data['bandgap-ML(eV)'] = y_pred
    os.chdir(predict_directory)
    data.to_csv('pred_gbr_4.csv',index=False)


def predict_plot(feature):
    data = pd.read_csv('pred_gbr_4.csv', header=0)
    x = data[feature]
    y = data['bandgap-ML(eV)']
    plt.scatter(x, y, color='gray', label='predict set')
    origin_data = pd.read_csv('HOIP-30_drop.csv', header=0)
    X_train, X_test, y_train, y_test = train_test_split(origin_data[feature], origin_data['bandgap-PBE(eV)'], test_size=0.20,
                                                        random_state=1)
    plt.scatter(X_train, list(y_train), color='blue', label='train set')
    plt.scatter(X_test, list(y_test), color='red', label='test set')
    plt.xlabel(feature)
    plt.ylabel('bandgap-ML(eV)')
    plt.axis([data[feature].min() * 0.90, data[feature].max() * 1.1, -0.5, 7.2])
    plt.legend(loc=4)
    plt.savefig(feature + '.png')
    plt.close()
    plt.show()


def predict_plot_byX(feature):
    data = pd.read_csv('pred_gbr_4.csv', header=0)
    X_list=['F', 'Cl', 'Br', 'I']
    X_separated_data = {}
    for x in X_list:
        X_separated_data[x] = data[data['X-site'] == x]
    color_list = {'F':'deepskyblue', 'Cl':'orange', 'Br':'gray', 'I':'yellow'}
    for Xsite in X_list:
        x = X_separated_data[Xsite][feature]
        y = X_separated_data[Xsite]['bandgap-ML(eV)']
        plt.scatter(x, y, color=color_list[Xsite], label=Xsite)
    plt.xlabel(feature)
    plt.ylabel('bandgap-ML(eV)')
    plt.axis([data[feature].min() * 0.90, data[feature].max() * 1.1, -0.5, 7.2])
    plt.legend(loc=4)
    plt.savefig(feature + '_X.png')
    plt.close()
    plt.show()