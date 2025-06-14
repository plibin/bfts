from argparse import ArgumentParser
import csv
import numpy as np
import scipy.stats as st
import pandas as pd

parser = ArgumentParser(description="merge_results")

parser.add_argument("-d", "--dir", dest="dir", type=str, required=True)
parser.add_argument("-a", "--algos", dest="algos", type=str, required=True)
parser.add_argument("-r", "--replicates", dest="replicates", type=int, required=True)
parser.add_argument("-T", "--horizon", dest="horizon", type=int, required=True)
parser.add_argument("-t", "--stat", dest="stat", type=str, required=True)

args = parser.parse_args()

#def confidence_interval(data, confidence=0.95):
#    a = 1.0 * np.array(data)
#    n = len(a)
#    se = st.sem(a)
#    h = se * st.t.ppf((1 + confidence) / 2., n-1)
#    return h

algos = args.algos.split(",")

for algo in algos:
    results = np.zeros((args.horizon, args.replicates))
    for r in range(args.replicates):
        csv_fn = args.dir + "/" + algo + "-" + str(r+1) + "." + args.stat
        print(csv_fn)
        with open (csv_fn) as csv_file:
            read_csv = csv.reader(csv_file, delimiter=',')
            next(read_csv)
            for row in read_csv:
                t = int(row[0])
                #AT-LUCB plays 2 arms (2 samples) per time step
                if algo == "atlucb":
                    assert(t <= args.horizon / 2)
                    v = row[1]
                    i = 2*(t-1)
                    results[i][r] = v
                    results[i+1][r] = v
                else:
                    assert(t <= args.horizon)
                    v = float(row[1])
                    results[t-1][r] = v
    
    h = list(range(1, args.horizon + 1)) 
    avg = np.apply_along_axis(np.mean, 1, results)
    var = np.apply_along_axis(np.var, 1, results)

    df = pd.DataFrame(data={'horizon':h, 'avg':avg, 'var':var})
    df.to_csv(args.dir + "/" + algo + "-" + args.stat + "-merged.csv")
