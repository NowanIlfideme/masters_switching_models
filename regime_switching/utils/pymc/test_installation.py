"""Run this module to check the installation of PyMC3."""

import logging

import numpy as np
import pandas as pd


def main():

    logging.basicConfig(level=logging.INFO)
    start = pd.Timestamp.now()

    # Import pymc3
    import pymc3 as pm

    # Create fake data

    rvs = np.random.default_rng(seed=1800)
    obs = rvs.normal(3, 3, size=20)

    # Create simple model

    model = pm.Model()
    with model:
        mu = pm.Normal("mu", mu=0, sd=10)
        sd = pm.HalfCauchy("sd", beta=3)
        y = pm.Normal("y", mu=mu, sd=sd, observed=obs)  # noqa

    # Fit it

    with model:
        trace = pm.sample(
            draws=500, tune=500, cores=1, chains=2, random_seed=42
        )

    trace  # noqa

    end = pd.Timestamp.now()
    elapsed = end - start
    logging.info(f"Elapsed: {elapsed}")

    if elapsed > pd.Timedelta("3 minutes"):
        logging.warning("Your PyMC3 install is very slow!")


if __name__ == "__main__":
    main()
