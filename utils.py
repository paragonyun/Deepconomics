from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

from sklearn.preprocessing import PowerTransformer



import pandas as pd


def diff(df, not_satisfied_lst: list , lag = 3) -> pd.DataFrame  :
    print('차분을 통해 계절성과 정상성을 제거합니다.')
    
    diff_df = df.copy()
    lst = not_satisfied_lst.copy()

    for col in lst :
        diff_df[col] = df[col].diff(lag)
        
    diff_df.dropna(inplace=True)
    return diff_df

def yeo_johnson(df, not_satisfied_lst : list) -> pd.DataFrame :
    
    boxcox_df = df.copy()
    lst = not_satisfied_lst.copy()

    for col in lst :
        boxcox = PowerTransformer(method='yeo-johnson')
        boxcox_df[col] = boxcox.fit_transform(boxcox_df[col].values.reshape(-1,1))

    boxcox_df.dropna(inplace=True)

    return boxcox_df


# def yj_transform(df, not_satisfied_lst:list) :
#     print('여존슨 변환을 통해 계절성과 정상성을 제거합니다.')

def check_stationality(df) :
    '''
    KPSS 테스트 : P-VALUE가 0.05보다 커야됨
    ADF 테스트 : P-VALUE가 0.05 보다 작아야됨
    '''
    ori_df = df.copy()
    
    out_lst = []

    kpass_cols = []
    adf_cols = []
    print('정상성을 체크합니다')
    for col in ori_df.columns :
        _, p_value_kpss, _, _ = kpss(ori_df[col].values)
        values = adfuller(ori_df[col].values)
        # print(f'Column : {col}')
        # print(f'\tKPSS p-value : {p_value_kpss:.2f}')
        # print(f'\tADF p-value : {values[1]:.2f}\n')
        kpass_cols.append(round(p_value_kpss,3))
        adf_cols.append(round(values[1],3))

        if p_value_kpss < 0.05 or values[1] >= 0.05 :
            out_lst.append(col)


    p_value_df = pd.DataFrame([kpass_cols, adf_cols], columns = ori_df.columns, index=['KPSS', 'ADF']).T
    
    TF_lst = [True if p_value_df.iloc[i, 0] >= 0.05 and p_value_df.iloc[i, 1] < 0.05 else False for i in range(len(p_value_df))]

    p_value_df['정상성_충족'] = TF_lst


    return out_lst, p_value_df





def _check_VIF(df, top_n) :
    vif_df = pd.DataFrame()
    vif_df['VIF_FACTOR'] = [vif(df.values, i) for i in range(df.shape[1])]

    vif_df['Feature_Name'] = df.columns

    fin_df = vif_df.sort_values(by='VIF_FACTOR', ascending=True)['Feature_Name'][:top_n].values

    return fin_df







