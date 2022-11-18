from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from xgboost import plot_importance
from lightgbm import plot_importance
# from shap import *
from eli5.sklearn import permutation_importance 

class xgb :
    '''
    XGB 모델을 돌리기 전 데이터는 정상성이 확보 되어 있는 상태여야 합니다!
    '''
    def __init__(self, X_train, y_train , X_test, y_test) :
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def train(self, Regressor='x') :
        if Regressor == 'x' :
            model = XGBRegressor()

        elif Regressor == 'l' :
            model = LGBMRegressor()

        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)

        return preds, model

    def plot_preds(self, preds) :
        
        plt.plot(self.y_test, label = 'Actual')
        plt.plot(preds, label = 'Prediction')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def check_importances(self, model) :
        fig, axes = plt.subplots(1,1, figsize=(10,25))

        plot_importance(model, ax=axes[0])
        
        plt.show()



















