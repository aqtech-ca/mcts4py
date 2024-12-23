#!/usr/bin/env python
from setuptools import setup, find_packages

_core_require = [ "numpy==1.23.1", "setuptools==70.2.0", "graphviz~=0.20.3"]

_tests_require = ["pytest", "hypothesis", "pytest-benchmark"]

_gym_require = ["gym[all]==0.26.2"]

setup(
    name="mcts4py",
    version='0.20.2',
    packages=find_packages(exclude="tests"),
    install_requires=_core_require,
    tests_require=_tests_require,
    extras_require={"tests": _tests_require, "drawing": ["python-igraph"], "gym_samples": _gym_require},
    author="Larkin Liu",
    author_email="larkin@aqtech.ca",
    description="Python implementation of Monte Carlo Tree Search (MCTS)",
    download_url="",
    license="MIT",
    url="https://github.com/aqtech-ca/mcts4py",
    zip_safe=True,
)

# https://gist.github.com/wjladams/f00d6c590a4384ad2a92bf9c53f6b794
# python3 -m pip install --index-url https://test.pypi.org/simple/ mcts4py-0.13
# pip install git+https://github.com/aqtech-ca/mcts4py@larkin/checks

# to pypi
# rm -r dist/*; python setup.py sdist bdist_wheel; twine upload --repository testpypi dist/*


