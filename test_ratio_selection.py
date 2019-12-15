import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as GBR

def ratio_test(cwdd):
    df = pd.DataFrame(columns=['ratio','score'])
    for i in range(1,100):
        X_train, X_test, y_train, y_test, predict_X, features = pre.raw_preprocessing(cwdd,i/100)
        reg = GBR(random_state = 1)
        reg.fit(X_train,y_train)
        df = df.append(pd.DataFrame([[1-i/100, reg.score(X_test,y_test)]],columns=df.columns), ignore_index=True)
    plt.plot(df['ratio'], df['score'], 'k.-')
    plt.xlabel('train_set_ratio')
    plt.ylabel('score')
    plt.savefig(cwdd + 'ratio_score.png')
    df.to_csv(cwdd+'ratio.png', index=None)
