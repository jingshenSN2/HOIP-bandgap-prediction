from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocessing(cwd):
    reg = MinMaxScaler()
    trates = pd.read_csv(cwd+'HOIP-30.csv',header=0)
    predic = pd.read_csv(cwd+'unknown_comb.csv',header=0)
    trates_y = trates['bandgap-PBE(eV)']
    trates.drop(['A-site','B-site','X-site','bandgap-PBE(eV)'], axis=1, inplace=True)
    predic.drop(['A-site','B-site','X-site','hash'], axis=1, inplace=True)
    dfall = trates.append(predic)
    dfall_mm = reg.fit_transform(dfall)
    trates_X = dfall_mm[0:203]
    predict_X = dfall_mm[203:-1]
    X_train, X_test, y_train, y_test = train_test_split(trates_X, trates_y, test_size=0.20, random_state=1)
    return X_train, X_test, y_train, y_test, predict_X

