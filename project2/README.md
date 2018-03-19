# Project 2

### CS 7642 Spring 2018

### Dan Frakes (dfrakes3)

#### Requirements

This code uses the following Python libraries:

* `numpy`
* `matplotlib`
* `keras`

If you're having trouble, or there's some leftover unused library that's still being imported somewhere that I'm forgetting, there is a `requirements.txt` file with a `pip freeze` of the virtualenv I used to create and run this project:

```bash
# create/activate your virtualenv if you're using one
pip install -r requirements.txt
```

#### Getting Started

To run Project 2 code, clone the repository, cd into `project2` subdirectory, and run `python main.py`.

By default, this script will run the code to train (continue training) the DQN agent.  I tried to include all hyperparameters as args using `argparse`, so see `main.py` or below for those options.  These should be self-explanatory, but otherwise there is help text for each arg:

```
parser.add_argument('env_id', nargs='?', default=DEFAULT_ENV, help='Select the environment to run')
parser.add_argument('-l', '--learner', default='dqn', choices=['q', 'dqn'], help='Select agent to train with')
parser.add_argument('--alpha', default=ALPHA, type=float, help='NN learning rate')
parser.add_argument('--gamma', default=GAMMA, type=float, help='Discount rate')
parser.add_argument('--epsilon', default=EPSILON, type=float, help='Random action rate')
parser.add_argument('--edr', default=EPSILON_DECAY_RATE, type=float, help='Epsilon decay rate')
parser.add_argument('--minepsilon', default=EPSILON_MIN, type=float, help='Minimum epsilon value')
parser.add_argument('--memlimit', default=MEMORY_LIMIT, type=int, help='DQN buffer size')
parser.add_argument('--nrf', default=NET_REPLACEMENT_FREQ, type=int, help='NN weight replacement frequency')
parser.add_argument('--batchsize', default=BATCH_SIZE, type=int, help='Mini-batch sampling size')
parser.add_argument('--maxepisodes', default=MAX_EPISODES, type=int, help='Mini-batch sampling size')
```

I spent most of my time adjusting and refactoring my DQN agent.  The `main.py` includes discretization if you're using the Q-Learner, but the majority of my work was spent on DQN.

#### Settings

Additionally, default hyperparameter values can be adjusted in `variables.py`.
