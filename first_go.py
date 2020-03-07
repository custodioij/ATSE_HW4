import pandas as pd
import numpy as np


data = pd.read_excel("USEMP.xlsx")
dt = data.Month
Mdata = np.array(data.iloc[:,1:])

# TODO: function to get the windows
# cf: https://stackoverflow.com/questions/22898824/filtering-pandas-dataframes-on-dates
# df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]
# dates[(dates >= '1980-01-01')]  #& (dates['Month'] < '2013-02-01')
# dates[(dates >= '1980-01-01')].index


def window(dates=dt, first_forecast="1980-01-01", window_length=None):
    # Overly complicated for such a simple task
    # last on will be 655, so that index 657 can be forecast and tested
    if window_length is None:
        window_length = dates[(dates == first_forecast)].index[0] - 1
    last_forecast = dates.index[-1]
    forecast_periods = last_forecast - window_length
    windows = []
    for i in range(forecast_periods):
        windows += [(last_forecast - forecast_periods - window_length + i,
                     last_forecast - forecast_periods + i)]
    return windows

def EstimateMM(Y, X):
    betas = np.linalg.inv(X.T @ X) @ X.T @ Y
    return betas


def ErrorsVec(Y, X, betas):
    err = Y - X @ betas
    return err


def EstSigma2(Y, X, betas):
    err = ErrorsVec(Y, X, betas)
    n = len(X)
    k = len(betas)
    sig2 = (err.T @ err)/(n - k)
    return sig2


def BetasSE(Y, X, betas):
    """
    Returns the standard errors of the estimated betas
    """
    sig2 = EstSigma2(Y, X, betas)
    BigSigma = sig2 * np.linalg.inv(X.T @ X)
    Bse = np.sqrt(np.diagonal(BigSigma))
    return Bse


# def kitchen_sink(X, Y):
