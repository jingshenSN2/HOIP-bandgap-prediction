import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os

def predict(model_directory, predict_directory):
    data = pd.read_csv('unknown_comb.csv', header=0)
    os.chdir(model_directory)
    gbr = joblib.load('gbr_4_0.9669.m')
    X_pred = data[['P_A', 'P_B', 'X_p-electron', 'VE_B']]
    y_pred = gbr.predict(X_pred)
    data['bandgap-ML(eV)']=y_pred
    os.chdir(predict_directory)
    data.to_csv('pred_gbr_4.csv',index=False)