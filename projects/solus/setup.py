\
"""
Minimal setup.py shim — delegates to pyproject.toml via setuptools.
Kept for tooling compatibility (pip < 24, older build frontends).
"""
from setuptools import setup
setup()
