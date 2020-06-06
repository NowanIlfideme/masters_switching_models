import numpy as np  # noqa
import pymc3 as pm
import theano.tensor as tt


class MarkovStateTransitions(pm.Categorical):
    """Distribution of Markov state transitions.
    
    Based on code from: 
    https://sidravi1.github.io/blog/2019/01/25/heirarchical-hidden-markov-model
    """

    def __init__(self, trans_prob=None, init_prob=None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)

        self.trans_prob = trans_prob
        self.init_prob = init_prob

        # Housekeeping
        self.mode = tt.cast(0, dtype="int64")
        self.k = 2

    def logp(self, x) -> tt.Tensor:
        """Log likelihood of chain `x`."""

        trans_prob = self.trans_prob

        p = trans_prob[
            x[:-1]
        ]  # probability of transitioning based on previous state
        x_i = x[1:]  # the state you end up in

        init_like = pm.Categorical.dist(p=self.init_prob).logp(x[0])
        chain_like = pm.Categorical.dist(p, shape=(self.shape[0], 2)).logp_sum(
            x_i
        )

        return init_like + chain_like

    # TODO: random() to allow sampling chains
