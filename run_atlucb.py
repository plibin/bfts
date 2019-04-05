import numpy as np
from argparse import ArgumentParser

import environments
from algorithms.atlucb import AT_LUCB 
from run_utils import print_header,run

parser = ArgumentParser(description="ATLUCB")

parser.add_argument("-s", "--seed", dest="seed", type=int, required=True)
parser.add_argument("-t", "--time", dest="time", type=int, required=True)
parser.add_argument("-m", "--m", dest="m",type=int, required=True)
parser.add_argument("-e", "--environment", dest="env", type=str, required=True)

args = parser.parse_args()

np.random.seed(args.seed)
(real_means, bandit) = environments.select(args.env)

print_header(args.m)
sigma=0.5
alpha=0.99
epsilon=0
algo = AT_LUCB(bandit, args.m, sigma, alpha, epsilon)
run(algo, args.time)
