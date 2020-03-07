import numpy as np


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


def OLS_predict(X_new, betas):
    """ To be tested! """
    return X_new @ betas