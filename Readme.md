# MCTS library in Python

## Requirements

- Python 3.9+

## Samples

To run samples:
- run `pip install -e .` in the root directory
- run `python main.py` in the sample directory

To install from PyPI,

```
python3 -m pip install --index-url https://test.pypi.org/simple/ mcts4py
```


## Running the MENTSsolver vs UCTsolver Comparison in Gym Environments

Install Requirements:

`python3 -m pip install -r requirements.txt`
`python3 -m pip install -r requirements_gym.txt`

Set the Python Path:

`export PYTHONPATH=/path/to/mcts4py:$PYTHONPATH`

Run the Comparisons:
- CartPole:  `python3 samples/gym/cartPole/CartPole.py`
- Frozen Lake:  `python3 samples/gym/frozenLake/FrozenLake.py`

The exploration_constant can be adjusted directly in the evaluate_solver method located in the respective scripts.