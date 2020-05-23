"""Regime-switching time series models."""

from pathlib import Path

__root__ = Path(__file__).parent.absolute()
__description__ = "Regime-switching time series models."

# Load current version.
try:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        # package is not installed
        __version__ = "not-installed"
