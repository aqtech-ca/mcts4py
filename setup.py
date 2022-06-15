#!/usr/bin/env python
from setuptools import setup, find_packages

_tests_require = ["pytest", "hypothesis", "pytest-benchmark"]

setup(
    name="mcts4py",
    version=0.0,
    packages=find_packages(exclude="tests"),
    tests_require=_tests_require,
    extras_require={"tests": _tests_require, "drawing": ["python-igraph"]},
    author="Larkin Liu",
    author_email="larkin@aqtech.ca",
    description="Python implementation of Monte Carlo Tree Search (MCTS)",
    classifiers=CLASSIFIERS,
    download_url="",
    license="MIT",
    url="",
    zip_safe=True,
)