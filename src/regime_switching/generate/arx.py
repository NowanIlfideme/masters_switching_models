

import numpy as np 
import pandas as pd 
from regime_switching.generate import SeriesGenerator 


class VARXGenerator(SeriesGenerator):
    """VARX model (n dimensions, k lags, m exogenous).

    This model assumes normality in the innovation process.
    
    Attributes
    ----------
    coef_ar : dict of array-like, {k : (n, n)}
        A coefficient matrix per each lag.
    coef_exog : array-like, (m, n)
        Exogenous coefficients. 
        TODO: Maybe also a dict, per lag?
    constants : array-like, (n)
        Constant values per component. Defaults to 0 values. 
    covariance : array-like, (n, n)
        Covariance matrix of residuals. Defaults to 0-matrix. 
    endogenous : pd.Index or None 
        Names of the endogenous to use. 
    exogenous : pd.Index or None 
        Names of the exogenous variables to use. 
    """

    def __init__(
        self, coef_ar={}, coef_exog=[], constants=0, covariance=0, 
        endogenous=None, exogenous=None, 
        random_state=None
    ):
        super().__init__(random_state=random_state) 

        if isinstance(endogenous, str):
            endogenous = [endogenous]

        if isinstance(exogenous, str):
            exogenous = [exogenous]

        self.coef_ar = {
            k: pd.DataFrame(v, index=endogenous, columns=endogenous) 
            for k, v in coef_ar.items()
        }

        self.coef_exog = pd.DataFrame(
            coef_exog, index=exogenous, columns=endogenous, 
        ) 

        self.constants = pd.Series(constants, index=endogenous) 
        self.covariance = pd.DataFrame(
            covariance, index=endogenous, columns=endogenous, 
        )

        # TODO: Deal with incorrect indexes being passed.

    @classmethod
    def random_model(
        cls, n, m=0, p_max=1, p_portion=1.0, 
        cov_min_rank=None, cov_max_rank=None, 
        random_state=None
    ):
        """Generates a model, based on parameters passed. 
        
        Parameters
        ----------
        n, m : int
            Number of endogenous and exogenous variables.
        p_max : int
            Maximum lag to use.
        p_portion : float
            What portion of the lags to use. To use all, set to 1.
        cov_min_rank, cov_max_rank : int or None
            Minimum, maximum ranks of the covariance matrix. 
            If None, sets to n. 
        random_state : None, int, np.RandomState
            Random state for `random_model`, not the resulting generator!
        """
        rng = cls._fix_rng(random_state)

        # Generate AR coefficients
        lags = np.arange(1, p_max + 1)[
            rng.binomial(1, p_portion, size=p_max).astype(bool)
        ]

        # HACK: Force stationarity! 
        # Currently doesn't, it's just a random matrix per coef
        coef_ar = {
            lag: rng.uniform(-0.5, 1, size=(n, n))
            for lag in lags
        }

        coef_exog = rng.uniform(0, 1, size=(m, n)) 
        constants = rng.uniform(-1, 1, size=(n)) 

        # Generate covariance matrix
        if cov_min_rank is None:
            cov_min_rank = n
        if cov_max_rank is None:
            cov_max_rank = n
        rank_L = rng.randint(cov_min_rank, cov_max_rank + 1)
        L = rng.uniform(-0.5, 1, size=(n, rank_L))
        covariance = L @ L.T 

        return cls(
            coef_ar=coef_ar, coef_exog=coef_exog, 
            constants=constants, covariance=covariance, 
            random_state=rng.randint(100000)
        )

    @property
    def endogenous(self):
        """Names of endogenous variables (components)."""
        return self.constants.index 

    @property
    def exogenous(self): 
        """Names of exogenous variables."""
        return self.coef_exog.index 

    @property
    def is_pure_ar(self):
        """True, if no exogenous part."""
        return len(self.exogenous) == 0 

    @property
    def lags(self):
        """All (endogenous) lags used in the model."""
        return sorted(self.coef_ar.keys()) 
    
    @property
    def coef_ar_df(self):
        """AR coeffs as a dataframe."""
        df1 = pd.concat([
            pd.concat([
                pd.Series(k, index=self.endogenous, name='lag'), 
                self.coef_ar[k]], axis='columns'
            )
            for k in self.coef_ar.keys()
        ], axis='rows')
        return df1.set_index(['lag', df1.index])

    def generate(self, index, exog=None, random_state=None):
        """Generates an independent chain for `index`.

        If not is_pure_ar, requires an exogenous set with the same index. 
        
        If `random_state` is not set, uses a copy of the internal state.
        """

        index = self._fix_index(index) 
        rng = self._fix_rng(random_state, self.random_state)

        if self.is_pure_ar:
            exog_impact = pd.DataFrame(0, index=index, columns=self.endogenous)
        else:
            if exog is None:
                raise ValueError(
                    "Is not pure AR, requires exogenous values. "
                    "Note that you can set a single value to propogate."
                ) 

            exog = pd.DataFrame(exog, index=index, columns=self.exogenous)
            exog_impact = pd.DataFrame(
                exog @ self.coef_exog, index=index, columns=self.endogenous)

        # Create answer
        
        # Create innovation process (IID normal errors) 
        eps = pd.DataFrame(
            rng.multivariate_normal(
                mean=np.zeros(len(self.endogenous)), 
                cov=self.covariance, 
                size=len(index)
            ),
            index=index,
            columns=self.endogenous
        )
        
        lags = self.lags
        maxlag = max(lags)

        # Find first values 
        # TEMP: use zeroes as initial values
        # TODO: Allow initial values 
        initial_values = pd.DataFrame(
            0, 
            index=pd.RangeIndex(-maxlag, 0), 
            columns=self.endogenous
        )

        res = pd.concat(
            [
                initial_values, 
                exog_impact + eps
            ], 
            axis='rows'
        ) 
        for i in range(maxlag, maxlag + len(index)): 
            ar_part = sum(self.coef_ar[l] @ res.iloc[i - l, :] for l in lags)
            res.iloc[i, :] += ar_part

        res = res.iloc[maxlag:, :]
        return res 


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    
    v1 = VARXGenerator(
        coef_ar={1: np.array([[0.5]]), }, 
        coef_exog=np.array([[1]]), 
        constants=[[0]], 
        covariance=np.array([[1]]), 
        endogenous=['endog'], exogenous=['exog']
    )
    v1.generate(100, 1).plot()
    plt.show()

    v2 = VARXGenerator(
        {1: [[1.001]], 2: [[-0.53]]}, 
        constants=10, 
        covariance=v1.covariance.values * 0.01, 
        # covariance=v.covariance * 0.01, 
        endogenous=['endog']
    )
    v2.generate(100).plot()
    plt.show()

    v3 = VARXGenerator(
        coef_ar={
            1: [[0.5, 0.6], [-0.4, 0.2]],
        },
        constants=0,
        covariance=[[0.8, 0.2], [0.2, 0.4]],
        endogenous=['a', 'b'],
    )
    v3.generate(100).plot()
    plt.show()

