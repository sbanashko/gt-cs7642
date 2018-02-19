"""
CS 7642 Sprint 2018
Project 1
Dan Frakes | dfrakes3
"""
import numpy as np

from project1.settings import *
from project1.temporal_difference.td import TD
from project1.utils import *


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
print '*** Figure 3 ***'
# Store episodes to repeatedly present
training_sets = [generate_episodes(NEPISODES, reset_states(), limit=6) for _ in range(NSETS)]

# Lambda values to test
lambda_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Record RMSE of converged TD for each lambda value
avg_errors = []

# Compute RMSE for 1 lambda value at a time
for lambda_val in lambda_vals:

    if DEBUG:
        print 'Running TD({})...'.format(lambda_val)

    # Record converged state value estimates for current lambda
    td_vals = []

    # Run 100 training sets using lambda value
    for training_set in training_sets:

        # Capture converged TD state value estimates
        td_val = TD(lambda_val, alpha=0.21, alpha_decay_rate=0.9, episodes=training_set, epsilon=1e-3)
        td_vals.append(td_val)

    # Capture RMSE
    avg_errors.append(rmse(td_vals, ACTUAL_STATE_VALUES))

if DEBUG:
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
print '*** Figure 4 ***'
lambda_vals = [0.0, 0.3, 0.8, 1.0]
alpha_vals = np.arange(0, 0.6, 0.01)
training_sets = [generate_episodes(NEPISODES, reset_states(), limit=6) for _ in range(NSETS)]

# Collect TD values for a single training set using each alpha value
avg_errors = []
best_alphas = []

for ld in lambda_vals:

    # Collect TD values as nested array for multiple plots
    ld_errors = []
    min_ld_error = 100
    best_alpha = 0

    for a in alpha_vals:

        # Record state value estimates for current lambda/alpha combination
        td_vals = []

        for training_set in training_sets:
            td_vals.append(TD(ld, alpha=a, max_iter=1, episodes=training_set))

        new_error = rmse(td_vals, ACTUAL_STATE_VALUES)
        ld_errors.append(new_error)
        if new_error < min_ld_error:
            min_ld_error = new_error
            best_alpha = a

    best_alphas.append(round(best_alpha, 2))
    avg_errors.append(ld_errors)

if DEBUG:
    print avg_errors
plot_alpha(alpha_vals, avg_errors, lambda_vals, xlab=u'$\\alpha$', legend=False)


'''Figure 4 experiment with more continuous lambda and alpha values'''
# for i in range(4, 21, 4):
#
#     if DEBUG:
#         print i
#
#     # Generate new set of episodes limiting episode length to i
#     tempsets = [generate_episodes(NEPISODES, reset_states(), limit=i) for _ in range(NSETS)]
#
#     # Collect TD values as nested array for multiple plots
#     temperrors = np.empty((0, 61))
#
#     for ld in np.linspace(0.0, 1.0, 21):
#
#         templderrors = []
#
#         for a in np.linspace(0, 0.6, 61):
#             tdv = []
#             for tset in tempsets:
#                 tdv.append(TD(ld, alpha=a, max_iter=1, episodes=tset))
#
#             templderrors.append(rmse(tdv, ACTUAL_STATE_VALUES))
#
#         temperrors = np.vstack([temperrors, templderrors])
#
#     fig4_frame(np.linspace(0.0, 1.0, 21), np.linspace(0, 0.6, 61), temperrors, i)


'''
Figure 5
Average error at best alpha value on random-walk problem. Each data point
represents the average over 100 training sets of the error in the estimates
found by TD(lambda), for particular lambda and alpha values, after a single presentation
of a training set. The lambda value is given by the horizontal coordinate. The alpha
value was selected from those shown in Figure 4 to yield the lowest error
for that lambda value.
'''
# Cache fig4 calculations of best alphas (Frakes)
# best_alphas = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
#                         0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
#                         0.2, 0.15, 0.15, 0.15, 0.15, 0.1,
#                         0.1, 0.1, 0.05])

# Copy of best alphas (Sutton)
# best_alphas = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
#                0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15,
#                0.15, 0.1, 0.1, 0.1, 0.05]
print '*** Figure 5 ***'

assert (len(lambda_vals) == len(best_alphas))

avg_errors = []

for i in range(len(lambda_vals)):

    td_vals = []

    for training_set in training_sets:
        td_vals.append(TD(lambda_vals[i], alpha=best_alphas[i], max_iter=1, episodes=training_set))

    # Capture RMSE
    avg_errors.append(rmse(td_vals, ACTUAL_STATE_VALUES))

if DEBUG:
    print avg_errors

plot(lambda_vals, avg_errors)
