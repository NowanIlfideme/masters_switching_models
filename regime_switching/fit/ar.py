import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions import draw_values

from regime_switching.generate.ar import VARXGenerator

# from pymc3.distributions.distribution import generate_samples
# from pymc3.distributions.shape_utils import to_tuple


class VAR(pm.Continuous):
    """Vector autoregressive process.

    .. math::

        y_t = TODO math

    Parameters
    ----------
    n_endog : int
        Number of endogenous variables (i.e. dimensionality of y).
    n_lag_endog : int
        Number of lags to estimate in the VAR process.
    t_const : pm.Distribution
        Shape: (n_endog,)
        Prior distribution for the constant.
        e.g. pm.Normal('t_const', 0, 10, shape=(n_endog,))
    t_ar : pm.Distribution
        Shape: (n_endog, n_endog, n_lag_endog)
        Prior distribution for AR coefficients.
        e.g. pm.Uniform('t_ar', -1, 1, shape=(n_endog, n_endog, n_lag_endog))
    t_init : pm.Distribution
        Shape: (n_lag_endog, n_endog)
        Prior distribution for initial values (before series starts).
        e.g. pm.Normal('t_init', 0, 10, shape=(n_lag_endog, n_endog))
    packed_chol : pm.Distribution
        Shape: (n_endog * (n_endog - 1)/ 2, )
        Packed representation of Cholesky decomposition.
        e.g. pm.LKJCholeskyCov('packed_chol', eta=ETA, n=n_endog, sd_dist=SD)

    TODO
    ----
    - Figure out n_endog from passed data?
    - Allow non-cholesky parametrization? Or even internalize it?
    - Rename parameters?
    """

    def __init__(
        self,
        n_endog,
        n_lag_endog,
        t_const,
        t_init,
        t_ar,
        packed_chol,
        dist_init=pm.Flat.dist(),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_endog = int(n_endog)
        self.n_lag_endog = int(n_lag_endog)

        self.t_const = tt.as_tensor_variable(t_const)
        self.t_init = tt.as_tensor_variable(t_init)
        self.t_ar = tt.as_tensor_variable(t_ar)

        # Covariance distribution args for innovation process
        # TODO: Maybe allow non-MvNormal innovation dist?

        self.packed_chol = packed_chol
        self.chol = chol = pm.expand_packed_triangular(
            n_endog, packed_chol, lower=True
        )
        self.cov = tt.dot(chol, chol.T)

        # Distribution for initial values - default is Flat a.k.a. "no idea"
        self.dist_init = dist_init

        # Test value
        self.mean = tt.as_tensor_variable(np.zeros(shape=(n_endog,)))

    @property
    def dist_innov(self) -> pm.Distribution:
        return pm.MvNormal.dist(mu=0, chol=self.chol)

    def logp(self, value: tt.Tensor) -> tt.TensorVariable:
        """Log probability, used in `pm.sample()`."""

        # Prepend initial values
        t_data = tt.as_tensor(value)
        try:
            t_data_pre = tt.concatenate([self.t_init, t_data], axis=0)
        except Exception:
            t_data_pre = self.t_init

        # Sum up total AR effect
        effect_ar = []
        for i in range(self.n_lag_endog):
            # lag is i+1
            _start = self.n_lag_endog - (i + 1)
            _end = -(i + 1)
            q = tt.dot(t_data_pre[_start:_end, :], self.t_ar[:, :, i])
            effect_ar.append(q)
        effect_ar = tt.add(*effect_ar)
        # effect_ar.shape.eval()  # == (T, n_endog)

        # Add constant to get errors
        expected_mean = effect_ar + self.t_const[None, :]
        errors = t_data - expected_mean
        # errors.shape.eval()  # == (T, n_endog)

        # Get likelihoods
        like_innov = self.dist_innov.logp(errors)
        # like_innov.shape.eval()  # == (T, )

        like_init = self.dist_init.logp(
            self.t_init
        )  # == (n_lag_endog, n_endog)?
        # like_init.shape.eval()  # == (n_lag_endog, n_endog) ?

        like_total = tt.sum(like_init) + tt.sum(like_innov)
        # like_total.shape.eval()  # == tuple()

        return like_total

    def random(self, point=None, size=None):
        """Random sample, used in `pm.sample_posterior_predictive()`."""

        a_const, a_init, a_ar, a_packed_chol, a_chol, a_cov = draw_values(
            [
                self.t_const,
                self.t_init,
                self.t_ar,
                self.packed_chol,
                self.chol,
                self.cov,
            ],
            point=point,
            size=size,
        )
        T = int(self.shape[0])

        def gen_1():
            my_varx_gen = VARXGenerator(
                endog=self.n_endog,
                lag_endog=self.n_lag_endog,
                coef_ar=a_ar,
                coef_covariance=a_cov,
                coef_const=a_const,
                # This seems correct - order is 'oldest' to 'newest':
                coef_initial_values=a_init,
                # Ignore nonstationary since these are samples:
                check_kwargs=dict(action_nonstationary="ignore"),
            )
            srs = my_varx_gen.generate(T)["output"].values
            return srs

        # TODO: what to do with `size`?
        random_samples = gen_1()
        return random_samples
