import numpy as np


def rmse(est, actual):
    """
    Helper method to calculate RMSE of TD values
    :param vals:
    :return:
    """
    # Create array (sets x states) of all state errors for all training sets
    errors = np.subtract(est, actual)

    # Calculate RMSE across state predictions for each state
    errors = np.sqrt(np.mean(pow(errors, 2), axis=1))

    # Average RMSEs across all training sets to return scalar TD error for each lambda
    return np.mean(errors)
