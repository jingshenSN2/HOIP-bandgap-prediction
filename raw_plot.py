import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def raw_feature_plot(feature):
    data = pd.read_csv('HOIP-30_drop.csv', header=0)
    X_train, X_test, y_train, y_test = train_test_split(data[feature], data['bandgap-PBE(eV)'], test_size=0.20,
                                                        random_state=1)
    plt.scatter(X_train, list(y_train), color='blue', label='train set')
    plt.scatter(X_test, list(y_test), color='red', label='test set')
    plt.xlabel(feature)
    plt.ylabel('bandgap-PBE(eV)')
    plt.axis([data[feature].min() * 0.95, data[feature].max() * 1.05, -0.5, 6.5])
    plt.legend(loc=0)
    plt.savefig('feature\\' + feature + '.png')
    plt.close()
    plt.show()
