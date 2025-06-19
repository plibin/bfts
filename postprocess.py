from argparse import ArgumentParser
import csv
import numpy as np

import environments

def postprocess(real_means_, m, stat, csv_fn):
    real_means = np.array(real_means_)
    real_m_top = np.argsort(-real_means)[:m]

    #print the output header
    print("t," + stat, flush=True)

    with open(csv_fn) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        header = next(read_csv)
        #first column is the time, the rest are the top arms
        for row in read_csv:
            time = int(row[0])
            m_top = list(map(int, row[1:1+args.m]))

            if args.stat == "min":
                min_ = np.min(real_means[m_top])
                print(str(time) + "," + str(min_))
            elif args.stat == "sum":
                sum_ = np.sum(real_means[m_top])
                print(str(time) + "," + str(sum_))
            elif args.stat == "prop_of_success":
                i = set(real_m_top).intersection(set(m_top))
                prop = len(i) / args.m
                print(str(time) + "," + str(prop))
            else:
                raise ValueError("Invalid statistic, choose from:" + \
                        "[min, sum, prop_of_success]")

if __name__ == "__main__":
    parser = ArgumentParser(description="postprocess")
    
    parser.add_argument("-c", "--csv_fn", dest="csv_fn", type=str, \
            required=True)
    parser.add_argument("-e", "--environment", dest="env", type=str, \
            required=True)
    parser.add_argument("-s", "--statistic", dest="stat", type=str, \
            required=True)
    parser.add_argument("-m", "--m", dest="m", type=int, required=True)

    args = parser.parse_args()

    (real_means, bandit) = environments.select(args.env)

    postprocess(real_means, args.m, args.stat, args.csv_fn)
