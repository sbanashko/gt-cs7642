from hw3.util import *

SAVE_JSON = True

T, R = construct_TR(30)

assert T.shape[1] == T.shape[2], 'Transition matrix does not have square action ndarray'
assert R.shape[1] == R.shape[2], 'Reward matrix does not have square action ndarray'

mdp_obj = construct_mdp_obj(T, R)
mdp_to_json(mdp_obj, SAVE_JSON)
