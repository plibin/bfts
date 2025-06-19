import numpy as np
from argparse import ArgumentParser

import environments
from algorithms.uniform import Uniform
from run_utils import print_header,run

def run_uniform(seed, bandit, m, time):
    np.random.seed(seed)

    print_header(args.m)
    algo = Uniform(bandit, m)
    run(algo, time)

if __name__ == "__main__":
    parser = ArgumentParser(description="Uniform m-top")

    parser.add_argument("-s", "--seed", dest="seed", type=int, required=True)
    parser.add_argument("-t", "--time", dest="time", type=int, required=True)
    parser.add_argument("-m", "--m", dest="m",type=int, required=True)
    parser.add_argument("-e", "--environment", dest="env", type=str, \
        required=True)

    args = parser.parse_args()

    (real_means, bandit) = environments.select(args.env)

    run_uniform(args.seed, bandit, args.m, args.time)
