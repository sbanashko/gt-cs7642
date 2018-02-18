"""
CS 7642 Sprint 2018
Project 1
Dan Frakes | dfrakes3
"""
import numpy as np

from plots import *
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
# # Store episodes to repeatedly present
# episodes = generate_episodes(NEPISODES, reset_states())
#
# lambda_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
#
# avg_errors = []
#
# print '*' * 80
#
# # Compute RMSE for 1 lambda value at a time
# for ld in lambda_vals:
#
#     print 'Running TD({})...'.format(ld)
#
#     # Record converged state value estimates for current lambda
#     td_vals = []
#
#     # Run 100 training sets using lambda value
#     for train_set in range(NSETS):
#
#         # Capture converged TD state value estimates
#         td_vals.append(TD(ld, alpha=0.05, episodes=episodes, epsilon=5e-4))
#
#         if (train_set + 1) % 10 == 0:
#             print 'Completed {} training sets'.format(train_set + 1)
#
#     # Capture RMSE
#     avg_errors.append(rmse(td_vals, ACTUAL_STATE_VALUES))
#
#     print '*' * 80
#
# print avg_errors
#
# plot(lambda_vals, avg_errors)

'''
Figure 4
Average error on random walk problem after experiencing 10 sequences.
All data are from TD(lambda) with different values of alpha and lambda. The dependent
measure is the RMS error between the ideal predictions and those found
by the learning procedure after a single presentation of a training set.
This measure was averaged over 100 training sets. The lambda = 1 data points
represent performances of the Widrow-Hoff supervised-learning procedure.
'''
lambda_vals = np.linspace(0.0, 1.0, 21)
# alpha_vals = np.linspace(0, 0.6, 13)
training_sets = [generate_episodes(NEPISODES, reset_states(), limit=6) for _ in range(NSETS)]
#
# # Collect TD values for a single training set using each alpha value
# errors = []
# best_alphas = []
#
# for ld in lambda_vals:
#
#     # Collect TD values as nested array for multiple plots
#     ld_errors = []
#     min_ld_error = 100
#     best_alpha = 0
#
#     for a in alpha_vals:
#         print u'TD({}) | alpha = {}...'.format(ld, a)
#
#         # Record state value estimates for current lambda/alpha combination
#         td_vals = []
#
#         for training_set in training_sets:
#             td_vals.append(TD(ld, alpha=a, max_iter=1, episodes=training_set))
#
#         new_error = rmse(td_vals, ACTUAL_STATE_VALUES)
#         ld_errors.append(new_error)
#         if new_error < min_ld_error:
#             min_ld_error = new_error
#             best_alpha = a
#
#     best_alphas.append(round(best_alpha, 2))
#     errors.append(ld_errors)
#
# print [l for l in lambda_vals]
# print [a for a in best_alphas]

# plot_alpha(alpha_vals, errors, lambda_vals, xlab=u'$\\alpha$', legend=False, file_counter=i)

# Uncomment below for fig4-animated GIF frames
# fig = plt.figure()
# for j in range(len(errors)):
#     plt.plot(alpha_vals, errors[j], label=lambda_vals[j])
# plt.xlim((-0.05, 0.65))
# plt.ylim((0.05, 0.75))
# plt.xlabel(u'$\\alpha$')
# plt.ylabel('ERROR')
# plt.text(0.4, 0.1, 'episode limit = {}'.format(i))
# fig.savefig(os.path.join('output', OUTPUT_DIR, '{}.png'.format(str((i - 4) / 2).zfill(4))))
# plt.close()

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
best_alphas = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.15, 0.15, 0.15, 0.15, 0.1,
                        0.1, 0.1, 0.05])

# Copy of best alphas (Sutton)
# best_alphas = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
#                0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15,
#                0.15, 0.1, 0.1, 0.1, 0.05]

assert (len(lambda_vals) == len(best_alphas))

# TESTING with only original fig 4 lambda vals
# fig4_indices = [0, 6, 16, 20]
# lambda_vals = [lambda_vals[i] for i in fig4_indices]
# best_alphas = [best_alphas[i] for i in fig4_indices]

avg_errors = []

for i in range(len(lambda_vals)):

    td_vals = []

    for training_set in training_sets:
        td_vals.append(TD(lambda_vals[i], alpha=best_alphas[i], max_iter=1, episodes=training_set))

    # Capture RMSE
    avg_errors.append(rmse(td_vals, ACTUAL_STATE_VALUES))

print avg_errors

plot(lambda_vals, avg_errors)

'''
Example 6.2 from Reinforcement Learning: An Introduction (Sutton & Barto, 1998), page 100.
'''
plot_value_updates()
plot_td_mc_comparison()
