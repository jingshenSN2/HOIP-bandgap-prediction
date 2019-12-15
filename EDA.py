import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def __boxplot(cwd, data, name):
    plt.figure(figsize=(18, 12))
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        data.iloc[:, i:i + 1].boxplot()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(cwd + name)


# TODO 这样画出的密度图对连续分布和离散分布的数据没有区分
def __density(cwd, data, name):
    plt.figure(figsize=(18, 12))
    for i in range(30):
        plt.ylabel(None)
        plt.subplot(5, 6, i + 1)
        if i == 29:
            continue
        else:
            data.iloc[:, i].plot(kind='kde', label=data.columns[i])
            plt.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(cwd + name)


def raw_data_describe(cwd):
    data = pd.read_csv(cwd + 'HOIP-30_drop.csv', header=0)
    data.drop(['A-site', 'B-site', 'X-site'], axis=1, inplace=True)
    data.describe(include=[np.number]).to_csv(cwd + 'eda\\raw_describe.csv')
    __boxplot(cwd, data, 'eda\\raw_boxplot.png')
    data.drop(['X_f-electron'], axis=1, inplace=True)
    __density(cwd, data, 'eda\\raw_density.png')


def pre_processing_data_describe(cwd):
    scaler = MinMaxScaler()
    data = pd.read_csv(cwd + 'HOIP-30_drop.csv', header=0)
    data.drop(['A-site', 'B-site', 'X-site'], axis=1, inplace=True)
    npdata = scaler.fit_transform(data)
    data = pd.DataFrame(npdata, columns=data.columns)
    data.describe(include=[np.number]).to_csv(cwd + 'eda\\pre_processing_describe.csv')
    __boxplot(cwd, data, 'eda\\pre_processing_boxplot.png')
    data.drop(['X_f-electron'], axis=1, inplace=True)
    __density(cwd, data, 'eda\\pre_processing_density.png')
