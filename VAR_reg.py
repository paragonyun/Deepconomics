## LIBs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

class VARModel :
    def __init__(self, df : pd.DataFrame) :
        self.df = df

    def find_p(self) :
        model = VAR(self.df)

        results = []
        for i in range(1, 10) :
            result = model.fit(i)
            results.append(result.aic)

        plt.plot(list(np.arange(1,10,1)), results)
        plt.title('AIC Scores')
        plt.xlabel('Order')
        plt.ylabel('AIC')
        plt.show()

        print(f'최소 AIC 값 : p = {min(results)}')
        print(f'최적의 p 값 : {results.index(min(results))+1}')

        return results.index(min(results))+1

    def train(self, p) :
        model = VAR(self.df)
        model_fitted = model.fit(p)

        print(model_fitted.summary())

        return model_fitted

    def forecast(self, fitted_model , train_df, test_df, min_p) :
        lagged = train_df.values[-min_p:]
        prediction = fitted_model.forecast(y = lagged, steps= 5) ## steps는 p값 보고 결정
        df_pred = pd.DataFrame(prediction, index=test_df.index, columns = test_df.columns+'_Pred')

        return df_pred