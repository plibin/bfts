import ast

from algorithms.posteriors.gaussian import Gaussian
from algorithms.posteriors.truncated_gaussian import TruncatedGaussian 
from algorithms.posteriors.dirichlet import Dirichlet 
from algorithms.posteriors.gamma import Gamma 
from algorithms.posteriors.beta import Beta 
from algorithms.posteriors.t_distribution import TDistribution 
from algorithms.posteriors.truncated_t_distribution import TruncatedTDistribution 

def parse_args(s):
    args_str = s[s.find("{"):s.find("}")+1]
    return ast.literal_eval(args_str)
    
def select(posterior_str):
    if posterior_str.startswith("gaussian{"):
        args = parse_args(posterior_str)
        var = float(args["var"])
        var0 = float(args["var0"])
        mu0 = float(args["mu0"])
        return Gaussian(var, var0, mu0)
    elif posterior_str.startswith("truncated_gaussian{"):
        args = parse_args(posterior_str)        
        var = float(args["var"])
        a = float(args["a"])
        b = float(args["b"])
        return TruncatedGaussian(a, b, var)
    elif posterior_str.startswith("truncated_t_distribution{"):
        args = parse_args(posterior_str)
        alpha = float(args["alpha"])
        a = float(args["a"])
        b = float(args["b"])
        return TruncatedTDistribution(alpha, a, b)
    elif posterior_str.startswith("t_distribution{"):
        args = parse_args(posterior_str)
        alpha = float(args["alpha"])
        return TDistribution(alpha)
    elif posterior_str.startswith( "dirichlet{"):
        args = parse_args(posterior_str)
        alpha = args["alpha"]
        cat = args["cat"]
        times_to_init = args["times_to_init"]
        return Dirichlet(alpha, cat, times_to_init)
    elif posterior_str.startswith( "gamma{"):
        args = parse_args(posterior_str)
        alpha = args["alpha"]
        beta = args["beta"]
        return Gamma(alpha, beta)
    elif posterior_str.startswith( "beta{"):
        args = parse_args(posterior_str)
        alpha = args["alpha"]
        beta = args["beta"]
        return Beta(alpha, beta)
    else:
        choices = \
            ["gaussian{var,var0,mu0}", "truncted_gaussian{var,a,b}", \
             "t_distribution{alpha}", "truncated_t_distribution{alpha,a,b}", \
             "dirichlet{[cat], [alpha]}", "gamma{alpha, beta}", \
             "beta{alpha, beta}"]
        raise ValueError("Invalid posterior \"" + posterior_str + "\", " + \
                "choose from " + \
                "[" + ", ".join(choices) + "]")
