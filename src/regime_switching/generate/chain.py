import numpy as np
import pandas as pd
from regime_switching.generate import SeriesGenerator


class IndependentChainGenerator(SeriesGenerator):
    """Independent chain in 1 dimension.
    
    Attributes
    ----------
    p : array-like 
        Selection probabilities.
    states : pd.Index or None 
        Names of the states to use. 
    """

    def __init__(self, p, states=None, random_state=None):
        super().__init__(random_state=random_state)

        p = pd.Series(p, index=states)
        if ((p < 0) | (p > 1)).any():
            raise ValueError("Selection probabilities must be in [0; 1].")
        if not np.allclose(1, p.sum()):
            raise ValueError("Selection probabilities must add to one.")
        self.p = p

    @property
    def states(self):
        return self.p.index

    def generate(self, index, random_state=None):
        """Generates an independent chain for `index`.
        
        If `random_state` is not set, uses a copy of the internal state.
        """
        index = self._fix_index(index)
        rng = self._fix_rng(random_state, self.random_state)

        # Create answer
        nms = rng.choice(self.states, size=len(index), p=self.p, replace=True)
        res = pd.Series(nms, index=index)
        return res

    @classmethod
    def make_random(cls, states=2, random_state=None):
        """Creates a random independent chain."""

        states = cls._fix_index(states)
        rng = cls._fix_rng(random_state)

        p = rng.uniform(0, 1, size=states.shape)
        p /= p.sum()
        return cls(p, states=states)


class MarkovChainGenerator(SeriesGenerator):
    """Markov chain in 1 dimension.
    
    Attributes
    ----------
    A : array-like
        Transition matrix. Must be 2D, square, rows must sum to 1.
    p : array-like or None 
        Initial probabilities. If None, defaults to uniform.
    states : pd.Index or None 
        Names of the states to use. 

    Note
    ----
    Could make a 2D Markov chain, using 3D A and 2D p, but that's out 
    of scope for this class. :)
    """

    def __init__(self, A, p=None, states=None, random_state=None):
        super().__init__(random_state=random_state)

        self.A = self._check_transition_matrix(A, states=states)

        states = self.A.index

        if p is None:
            n1 = 1 / len(states)
            self.p = pd.Series(n1, index=states)
        else:
            p = pd.Series(p, index=states)
            if ((p < 0) | (p > 1)).any():
                raise ValueError("Initial probabilities must be in [0; 1].")
            if not np.allclose(1, p.sum()):
                raise ValueError("Initial probabilities must add to one.")
            self.p = p

    @property
    def states(self):
        return self.A.index

    def generate(self, index, random_state=None):
        """Generates a Markov chain for `index`.
        
        If `random_state` is not set, uses a copy of the internal state.
        """
        index = self._fix_index(index)
        rng = self._fix_rng(random_state, self.random_state)

        # Create answer
        res = pd.Series(index=index)
        res.iloc[0] = rng.choice(self.states, p=self.p)
        for i in range(1, len(index)):
            p_i = self.A.loc[res.iloc[i - 1], :]
            res.iloc[i] = rng.choice(self.states, p=p_i)
        return res

    @classmethod
    def make_random(cls, states=2, random_state=None):
        """Creates a random Markov chain."""

        states = cls._fix_index(states)
        rng = cls._fix_rng(random_state)

        n = len(states)

        A = rng.uniform(0, 1, size=(n, n))
        A /= A.sum(axis=1)[:, np.newaxis]

        p = rng.uniform(0, 1, size=n)
        p /= p.sum()

        return cls(A, p=p, states=states)

    # Helper methods

    @staticmethod
    def _check_transition_matrix(A, states=None):
        """Returns proper transition matrix from `A` and `states` labels."""
        # Set transition matrix
        A_ = pd.DataFrame(A, index=states, columns=states)
        # Check squareness
        m, n = A_.shape
        if m != n:
            raise ValueError("Transition matrix must be square.")
        # Check if indexes are actually the same
        if not np.all(A_.index == A_.columns):
            raise ValueError("Index and columns don't match.")
        # Check if transition matrix sums to 1, with positive elements.
        if ((A_ < 0) | (A_ > 1)).any().any():
            raise ValueError("Transition matrix elements must be in [0; 1].")
        tot_probs = A_.sum(axis="columns")
        if not np.allclose(1, tot_probs):
            raise ValueError("Transition matrix rows must sum to 1.")
        return A_


# Useful shortcuts

ICG = IndependentChainGenerator
MCG = MarkovChainGenerator
