#!\usr\bin\python
"""Setuptools-based installation."""

from codecs import open  # To use a consistent encoding
from os import path

from setuptools import find_packages, setup

description = "Regime-Switching Time Series Models."
try:
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    long_description = description

#

setup(
    name="regime_switching",
    packages=find_packages(),
    description=description,
    long_description=long_description,
    author="Anatoly Makarevich",
    use_scm_version={"root": ".", "relative_to": __file__},
    setup_requires=["setuptools_scm"],
    install_requires=["Deprecated"],
)
