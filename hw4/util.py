import os
import sys


def save_txt(env, output_dir, episode_dir, iteration):
    stdout = sys.stdout
    new_file = os.path.join('output', output_dir, episode_dir, '{}.txt'.format(str(iteration).zfill(4)))
    sys.stdout = open(new_file, 'w')
    env.render()
    sys.stdout = stdout
