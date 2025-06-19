import numpy as np
from argparse import ArgumentParser

import environments
from algorithms.atlucb import AT_LUCB 
from run_utils import print_header,run

def run_atlucb(seed, bandit, m, time):
    np.random.seed(seed)

    print_header(m)
    sigma=0.5
    alpha=0.99
    epsilon=0
    algo = AT_LUCB(bandit, m, sigma, alpha, epsilon)
    run(algo, time)

if __name__ == "__main__":
    parser = ArgumentParser(description="ATLUCB")

    parser.add_argument("-s", "--seed", dest="seed", type=int, required=True)
    parser.add_argument("-t", "--time", dest="time", type=int, required=True)
    parser.add_argument("-m", "--m", dest="m",type=int, required=True)
    parser.add_argument("-e", "--environment", dest="env", type=str, \
            required=True)

    args = parser.parse_args()
    
    (real_means, bandit) = environments.select(args.env)

    run_atlucb(args.seed, bandit, args.m, args.time)
