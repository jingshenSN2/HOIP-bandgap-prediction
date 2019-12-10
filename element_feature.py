import pandas as pd
import os

def element_feature(cwd):
    df = pd.read_csv(cwd+'HOIP-30.csv',header=0)
    dfA = df[['A-site','r_A.eff','P_A', 'A_HOMO', 'A_LUMO']]
    dfB = df[['B-site','χ_B', 'r_B_s+p','r_B', 'IE_B', 'P_B', 'EA_B','IC_B', '1st_IP_B', 'VE_B', 'B_d-electron', 'B_p-electron', 'B_s-electron','B_f-electron']]
    dfX = df[['X-site','X_p-electron','1st_IP_X', 'χ_X', 'P_X', 'IC_X', 'r_X_s+p', 'r_X', 'X_s-electron','EA_X','X_d-electron', 'X_f-electron']]
    dfA1 = dfA.drop_duplicates('A-site').sort_values('r_A.eff')
    dfB1 = dfB.drop_duplicates('B-site').sort_values('B-site')
    dfX1 = dfX.drop_duplicates('X-site').sort_values('IC_X')
    dfA1.to_csv(cwd + 'A_feature.csv',index=False)
    dfB1.to_csv(cwd + 'B_feature.csv',index=False)
    dfX1.to_csv(cwd + 'X_feature.csv',index=False)