from argparse import ArgumentParser
import csv
import numpy as np
import scipy.stats as st
import pandas as pd

def merge(algos, horizon, replicates, dir_, stat):
    for algo in algos:
        results = np.zeros((horizon, replicates))
        for r in range(replicates):
            csv_fn = dir_ + "/" + algo + "-" + str(r+1) + "." + stat
            print(csv_fn)
            with open (csv_fn) as csv_file:
                read_csv = csv.reader(csv_file, delimiter=',')
                next(read_csv)
                for row in read_csv:
                    t = int(row[0])
                    #AT-LUCB plays 2 arms (2 samples) per time step
                    if algo == "atlucb":
                        assert(t <= horizon / 2)
                        v = row[1]
                        i = 2*(t-1)
                        results[i][r] = v
                        results[i+1][r] = v
                    else:
                        assert(t <= horizon)
                        v = float(row[1])
                        results[t-1][r] = v
    
        h = list(range(1, horizon + 1)) 
        avg = np.apply_along_axis(np.mean, 1, results)
        var = np.apply_along_axis(np.var, 1, results)

        df = pd.DataFrame(data={'horizon':h, 'avg':avg, 'var':var})
        df.to_csv(dir_ + "/" + algo + "-" + stat + "-merged.csv")

if __name__ == "__main__":
    parser = ArgumentParser(description="merge_results")
    
    parser.add_argument("-d", "--dir", dest="dir", type=str, required=True)
    parser.add_argument("-a", "--algos", dest="algos", type=str, required=True)
    parser.add_argument("-r", "--replicates", dest="replicates", type=int, \
            required=True)
    parser.add_argument("-T", "--horizon", dest="horizon", type=int, \
            required=True)
    parser.add_argument("-t", "--stat", dest="stat", type=str, required=True)

    args = parser.parse_args()

    algos = args.algos.split(",")
    merge(algos, args.horizon, args.replicates, args.dir, args.stat)

