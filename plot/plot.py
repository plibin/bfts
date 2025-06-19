from argparse import ArgumentParser
import csv
import numpy as np
import scipy.stats as st 
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import ticker 

def clean_algo_name(algo):
    if algo == "uniform":
        return "Uniform"
    elif algo == "atlucb":
        return "AT-LUCB"
    elif algo == "bfts":
        return "BFTS"

def clean_stat_name(stat):
    if stat == "sum":
        return r'$\sum_{i \in \mathcal{J}(t)} \mu_i$'
    elif stat == "prop_of_success":
        return r'$|\mathcal{J}(t) \cap \mathcal{J}^{*}|/m$'

def x_axis_fmt(x, pos): 
    s = '%d' % (x / 10**4)
    return s

def plot(algos, samples, dir_, stat, ymin, ymax, output):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
   
    #use latex for text rendering
    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')
    
    x = list(range(1, samples + 1))
    for algo in algos:
        df = pd.read_csv(dir_ + "/" + algo + "-" + stat + "-merged.csv")
        avg = df['avg']
        var = df['var']

        color = colors[algos.index(algo)]
        pl.fill_between(x, avg-var, avg+var, color=color, alpha=0.5)
        pl.plot(x, avg, color, label=clean_algo_name(algo))

    pl.tick_params(labelsize=16)

    ax = pl.gca()
    ax.xaxis.set_major_locator(pl.MaxNLocator(8))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_axis_fmt))

    pl.ylabel(clean_stat_name(stat), fontsize=16)
    pl.xlabel(r'\# of samples $\times 10^4$', fontsize=16)

    pl.legend(loc='lower right', fontsize='x-large')
    if ymin:
        pl.ylim(ymin=ymin)
    if ymax:
        pl.ylim(ymax=ymax)

    pl.tight_layout()

    if output is not None:
        #up the DPI for png
        if output.endswith(".png"):
            pl.savefig(output, bbox_inches='tight', pad_inches = 0.05, \
                       dpi=600)
        elif output.endswith(".pdf"):
            pl.savefig(output)
        elif output.endswith(".eps"):
            pl.savefig(output, format='eps', dpi=1000)
        else:
            sys.exit("Unsupported output format! Exit-ing")
    else:
        pl.show()

if __name__ == "__main__":
    parser = ArgumentParser(description="plot")

    parser.add_argument("-d", "--dir", dest="dir", type=str, required=True)
    parser.add_argument("-a", "--algos", dest="algos", type=str, required=True)
    parser.add_argument("-t", "--stat", dest="stat", type=str, required=True)
    parser.add_argument("-s", "--samples", dest="samples", type=int, \
            required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, \
            required=False)
    parser.add_argument("--ymin", dest="ymin", type=float, required=False)
    parser.add_argument("--ymax", dest="ymax", type=float, required=False)
    
    args = parser.parse_args()

    output = None
    if args.output:
        output = args.output

    algos = args.algos.split(",")
     
    plot(algos, args.samples, args.dir, args.stat, args.ymin, args.ymax, output)
