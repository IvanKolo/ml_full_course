import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(12345)

arparams = np.array([0.75, -0.25])
maparams = np.array([0.65, 0.35])

print(arparams)
print(maparams)

arparams = np.r_[1, -arparams]
maparams = np.r_[1, maparams]

nobs = 250
y = arma_generate_sample(arparams, maparams, nobs)
print(y)

dates = pd.date_range("1980-1-1", freq="M", periods=nobs)
y = pd.Series(y, index=dates)
arma_mod = ARIMA(y, order=(2, 0, 2), trend="n")
arma_res = arma_mod.fit()

print(arma_res.summary())

print(y.tail())


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(arma_res, start="1999-06-30", end="2001-05-31", ax=ax)
legend = ax.legend(loc="upper left")

plt.show()



### Hodrick-Prescott filter

import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

dta = sm.datasets.macrodata.load_pandas().data

index = pd.Index(sm.tsa.datetools.dates_from_range("1959Q1", "2009Q3"))
print(index)

dta.index = index
del dta["year"]
del dta["quarter"]
print(dta.head(10))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
dta.realgdp.plot(ax=ax)
legend = ax.legend(loc="upper left")
legend.prop.set_size(20)
plt.show()