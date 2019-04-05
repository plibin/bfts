# Boundary Focused Thompson sampling
This repository contains a python implementation of the "Boundary Focused Thompson sampling" (BFTS) algorithm. It also contains the bandit environments and algorithms (uniform sampling and AT-LUCB) used in the experiments of the "Bayesian Anytime m-top Exploration" manuscript.

The algorithms can be run by using the following scripts: run_uniform.py, run_atlucb.py, run_bfts.py.
These scripts all accept the following parameters:
- m: the top arms to consider
- time: the number of time steps the algorithm should run
- seed: the seed to initialise the random number generator, to make the runs reproducible
- env: the bandit environment on which to run the algorithm

The following environments are available: 
- Gaussian with fixed variance and linear means: linear
- Gaussian with fixed variance and polynomial means: polynomial
- Cartoon caption contest: captions
- Olivier's Poisson environment: poisson_olivier
- scaled Gaussian, inspired by epidemiological models: scaled_gaussian 

These environments can be parameterized using the python dictionary notation. The linear environment has for example 2 parameters: the number of arms (n) and the variance (var). For example, to denote a linear environment with 100 arms and a variance of .25, use "linear{'n':100, 'var':.25}".

A complete list of the bandit environments and their parameters:
- linear: number of arms (n), variance (var)
- polynomial: number of arms (n), variance (var)
- captions: number of arms (n)
- poisson_olivier: number of arms (n)
- scaled_gaussian: number of arms (n)

The BFTS algorithm expects an additional parameter: the posterior. The following posteriors are available, with their parameters between accolades:
- truncated Gaussian: truncated_gaussian{var, a, b}
- truncated t-distribution: truncated_t_distribution{alpha, a, b}
- Dirichlet: dirichlet{alpha, cat, times_to_init}
- Gamma: gamma{alpha, beta}

As for the environments, the parameters can be passed using the python dictionary notation. For example, to denote a Gaussian truncated on [0,1] with variance .25, use "truncated_gaussian{'var':.25,'a':0,'b':1}".

As an example, say we want to run 10000 time steps in the linear environment with a variance of .25 and 100 arms, and we want to investigate the 5 best arms.

For ATLUCB, we can use:
```
python run_atlucb.py -s 1 -t 10000 -m 5 -e "linear{'var':.25, 'n':100}" 
```

For BFTS, given that we use a truncated Gaussian posterior, we have: 
```
python run_bfts.py -s 1 -t 10000 -m 5 -e "linear{'var':.25, 'n':100}" -p "truncated_gaussian{'var':.25,'a':0,'b':1}"
```
