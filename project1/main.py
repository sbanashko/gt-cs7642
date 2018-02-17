"""
CS 7642 Sprint 2018
Project 1
Dan Frakes | dfrakes3
"""
import numpy as np

from plots import *
from project1.settings import *
from project1.temporal_difference.td import TD

'''
Figure 3
Average error on the random-walk problem under repeated presentations.
All data are from TD(lambda) with different values of lambda. The dependent measure
used is the RMS error between the ideal predictions and those found by
the learning procedure after being repeatedly presented with the training
set until convergence of the weight vector. This measure was averaged over
100 training sets to produce the data shown. The lambda = 1 data point is
the performance level attained by the Widrow-Hoff procedure. For each
data point, the standard error is approximately cr = 0.01, so the differences
between the Widrow-Hoff procedure and the other procedures are highly
significant.
'''
lambda_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

avg_errors = []

print '*' * 80

# Compute RMSE for 1 lambda value at a time
for ld in lambda_vals:

    print 'Running TD({})...'.format(ld)

    # Record converged state value estimates for current lambda
    td_vals = []

    # Run 100 training sets using lambda value
    for train_set in range(NSETS):

        # Capture converged TD state value estimates
        td_vals.append(TD(ld, num_episodes=NEPISODES))

        if (train_set + 1) % 10 == 0:
            print 'Completed {} training sets'.format(train_set + 1)

    # Create array (sets x states) of all state errors for all training sets
    td_errors = np.subtract(td_vals, ACTUAL_STATE_VALUES)

    # Calculate RMSE across state predictions for each state
    td_errors = np.sqrt(np.mean(pow(td_errors, 2), axis=1))

    # Average RMSEs across all training sets to return scalar TD error for each lambda
    avg_errors.append(np.mean(td_errors))

    print '*' * 80

print avg_errors

plot(lambda_vals, avg_errors)

'''
Figure 4
Average error on random walk problem after experiencing 10 sequences.
All data are from TD(lambda) with different values of alpha and lambda. The dependent
measure is the RMS error between the ideal predictions and those found
by the learning procedure after a single presentation of a training set.
This measure was averaged over 100 training sets. The lambda = 1 data points
represent performances of the Widrow-Hoff supervised-learning procedure.
'''

# plot()

'''
Figure 5
Average error at best alpha value on random-walk problem. Each data point
represents the average over 100 training sets of the error in the estimates
found by TD(lambda), for particular lambda and alpha values, after a single presentation
of a training set. The lambda value is given by the horizontal coordinate. The alpha
value was selected from those shown in Figure 4 to yield the lowest error
for that lambda value.
'''
# plot()

'''
Example 6.2 from Reinforcement Learning: An Introduction (Sutton & Barto, 1998), page 100.
'''
plot_value_updates()
plot_td_mc_comparison()
