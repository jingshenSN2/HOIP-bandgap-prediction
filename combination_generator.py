import pandas as pd
import math

def combination_generator(cwd):
    dfA = pd.read_csv(cwd+'A_feature.csv',header=0)
    dfB = pd.read_csv(cwd+'B_feature.csv',header=0)
    dfX = pd.read_csv(cwd+'X_feature.csv',header=0)

    df = pd.read_csv(cwd+'HOIP-30.csv',header=0)
    df.drop(df.index, inplace=True)
    df.drop('bandgap-PBE(eV)', axis=1, inplace=True)
    for ia in range(len(dfA)):
        la = dfA.iloc[ia]
        for ib in range(len(dfB)):
            lb = dfB.iloc[ib]
            for ix in range(len(dfX)):
                lx = dfX.iloc[ix]
                ra = la['r_A.eff']
                rb = lb['r_B']
                rx = lx['r_X']
                Tf = (ra + rx)/(math.sqrt(2)*(rb + rx))
                Of = rb / rx
                newRow = pd.DataFrame(
                [[la['A-site'],
                 lb['B-site'],
                 lx['X-site'],
                 Tf,
                 Of,
                 la['r_A.eff'],
                 la['P_A'],
                 la['A_HOMO'],
                 lb['χ_B'],
                 lb['r_B_s+p'],
                 la['A_LUMO'],
                 lb['r_B'],
                 lb['IE_B'],
                 lb['P_B'],
                 lb['EA_B'],
                 lx['X_p-electron'],
                 lb['IC_B'],
                 lb['1st_IP_B'],
                 lb['VE_B'],
                 lb['B_d-electron'],
                 lx['1st_IP_X'],
                 lx['χ_X'],
                 lx['P_X'],
                 lx['IC_X'],
                 lx['r_X_s+p'],
                 lx['r_X'],
                 lx['X_s-electron'],
                 lb['B_p-electron'],
                 lx['EA_X'],
                 lb['B_s-electron'],
                 lx['X_d-electron'],
                 lb['B_f-electron'],
                 lx['X_f-electron']
                 ]],columns=df.columns)
                df = df.append(newRow,ignore_index=True)
    df.to_csv(cwd+'all_combination.csv',index=False)

