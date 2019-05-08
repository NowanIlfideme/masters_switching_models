"""Module for fitting a single structural break."""

# import numpy as np
import pandas as pd
import statsmodels.api as sm

import typing
import traceback


def max_joint_likelihood(
    y: typing.Union[pd.DataFrame, pd.Series],
    fit_func: typing.Callable,
    buffer: int = 10,
) -> tuple:
    """Estimates break point by maximizing the joint likelihood of the models. 
    
    Parameters
    ----------
    y : pd.DataFrame or pd.Series
        Input series.
    fit_func : Callable
        Function that fits on a series and returns a model with the `llf` attribute.
    buffer : int
        Buffer at front, end to minimize "jumping".

    Returns
    -------
    break_point : object
        The estimated break point. Taken from original index.
    df : pd.DataFrame
        Log likelihoods of left (0), right (1), and `joint` models. 
        Index is an integer index by location.
    fails : pd.Series
        Series of exceptions during model fitting. Index is integer index by loc.
    """

    fails = {}

    idx = []
    lla = []
    llb = []
    for T in range(buffer, len(y.index) - buffer):
        try:
            l1 = fit_func(y.iloc[:T]).llf
            l2 = fit_func(y.iloc[T:]).llf
            lla.append(l1)
            llb.append(l2)
            idx.append(T)
        except Exception:
            fails[T] = traceback.format_exc()
    df = pd.DataFrame({0: lla, 1: llb}, index=idx)
    df["joint"] = df[0] + df[1]

    # TODO: Possibly smarter version?
    break_i = df["joint"].idxmax()
    break_point = y.index[break_i]

    fails = pd.Series(fails)

    return (break_point, df, fails)


def fit_ar(max_lags: int = 1) -> typing.Callable:
    """Returns fit function for AR.
    
    Returns
    -------
    inner : function
        Fits an AR model with given maximum number of lags.
    """

    def inner(y):
        return sm.tsa.AR(y).fit(maxlag=max_lags)

    return inner


def fit_var(max_lags: int = 1) -> typing.Callable:
    """Returns fit function.
    
    Returns
    -------
    inner : function
        Fits a VAR model with given maximum number of lags.
    """

    def inner(y):
        return sm.tsa.VAR(y).fit(maxlags=max_lags)

    return inner
