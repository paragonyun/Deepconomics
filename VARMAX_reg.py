from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


class VARMaxModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_order(self, train_df: pd.DataFrame):
        model = VAR(train_df)
        sorted_order = model.select_order(maxlags=12)

        print("최소값이 제일 많은 order를 선택하세요")
        print(sorted_order.summary())

    def train(self, train_df: pd.DataFrame, order):
        model = VARMAX(train_df, order=(order, 0), enforce_stationarity=True)
        fitted_model = model.fit(disp=True)
        print(fitted_model.summary())

        return fitted_model

    def forecast(self, fitted_model, train_df):
        predict = fitted_model.get_prediction(
            start=len(train_df), end=len(train_df) + 10 - 1
        )
        predictions = predict.predicted_mean

        predictions.columns = [str(col) for col in train_df.columns]

        return predictions

    def plot_predictions(self, preds, actuals):
        plt.plot(list(range(10)), preds, label="Prediction")
        plt.plot(list(range(10)), actuals, label="Actuals")

        plt.legend()
        plt.show()
