#!/usr/bin/env python
from setuptools import setup, find_packages

_tests_require = ["pytest", "hypothesis", "pytest-benchmark"]

setup(
    name="mcts4py",
    version=0.12,
    packages=find_packages(exclude="tests"),
    tests_require=_tests_require,
    extras_require={"tests": _tests_require, "drawing": ["python-igraph"]},
    author="Larkin Liu",
    author_email="larkin@aqtech.ca",
    description="Python implementation of Monte Carlo Tree Search (MCTS)",
    download_url="",
    license="MIT",
    url="https://github.com/aqtech-ca/mcts4py",
    zip_safe=True,
)

# https://gist.github.com/wjladams/f00d6c590a4384ad2a92bf9c53f6b794
# python3 -m pip install --index-url https://test.pypi.org/simple/ mcts4py-0.12