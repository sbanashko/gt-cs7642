# Project 3

### CS 7642 Spring 2018

### Dan Frakes (dfrakes3)

#### Requirements

This code uses the following Python libraries:

* `cvxopt`
* `glpk`
* `matplotlib`
* `numpy`

If you're having trouble, or there's some leftover unused library that's still being imported somewhere that I'm forgetting, there is a `requirements.txt` file with a `pip freeze` of the virtualenv I used to create and run this project:

```bash
# create/activate your virtualenv if you're using one
pip install -r requirements.txt
```

#### Getting Started

To run Project 3 code, clone the repository, cd into `project3` subdirectory, and run `python main.py`.

By default, this script will train a uCE-Q Learner against a random agent.  You can change the player and/or opponent learner type to one of the following:

* `QLearner`
* `FriendQLearner`
* `FoeQLearner`
* `CEQLearner`
* `RandomAgent`

Hyperparameter adjustments are best set in the agent constructor methods (see `agents.py`). Trackers like Q updates, random action rates, etc. collect data during training, so you will need to dig into the body of `main.py` to adjust or turn off these data gatherers. 

#### Settings

There are additional configurations set up in `vars.py`.
