from setuptools import setup, find_packages

from regime_switching import __version__

setup(
    name="regime_switching",
    description="Regime-switching time series models",
    version=__version__,
    author="Anatoly Makarevich",
    author_email="anatoly_mak@yahoo.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn", "statsmodels"],
    # TODO: Add extras and such
    # Testing (Nose tests)
    # Docs (Sphinx)
    # Optional dependencies?
)
