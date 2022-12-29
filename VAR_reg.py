## LIBs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


class VARModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def find_p(self):
        model = VAR(self.df)

        results = []
        for i in range(1, 20):
            result = model.fit(i)
            results.append(result.aic)

        plt.plot(list(range(len(results))), results)
        plt.title("AIC Scores")
        plt.xlabel("Order")
        plt.ylabel("AIC")
        plt.show()

        print(f"최소 AIC 값 : p = {min(results)}")
        print(f"최적의 p 값 : {results.index(min(results))+1}")

        return results.index(min(results)) + 1

    def train(self, p):
        model = VAR(self.df)
        model_fitted = model.fit(p)

        print(model_fitted.summary())

        return model_fitted

    def forecast(self, fitted_model, train_df, test_df, min_p, step=5):
        lagged = train_df.values[-min_p:]
        # lagged = test_df.values
        prediction = fitted_model.forecast(
            y=lagged, steps=step
        )  ## steps : X_test가 5개로 할 거니 5개로 결정...!
        cols = [str(i) for i in range(len(train_df.columns) - 1)]
        cols.append("Target_preds")
        df_pred = pd.DataFrame(prediction, index=test_df.index, columns=cols)

        return df_pred

    def grangers_causation_matrix(self, data, variables, test="ssr_chi2test"):
        """
        grangers_causation_matrix(df, variables = df.columns)
        """
        maxlag = 14

        df = pd.DataFrame(
            np.zeros((len(variables), len(variables))),
            columns=variables,
            index=variables,
        )
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(
                    data[[r, c]], maxlag=maxlag, verbose=False
                )
                p_values = [
                    round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)
                ]

                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + "_x" for var in variables]
        df.index = [var + "_y" for var in variables]
        return df

    def plot_impulse(self, fitted_model, num):
        fitted_model.irf(num).plot(response="Target", figsize=(10, 50))
        plt.tight_layout()
        plt.show()
