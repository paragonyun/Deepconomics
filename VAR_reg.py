## LIBs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

data = ...

data.index = ...

## Stationary Check
adf_test = adfuller('targetdata', autolag='AIC')


## differencing 
data_diff = data.diff().dropna()
adf_test = adfuller('targetdata', autolag='AIC') ## Re-check


## Model test


