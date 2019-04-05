def print_header(m):
    header = ['m %i' % i for i in range(1, m + 1)]
    print("t," + ",".join(header), flush=True)
    
def run(algo, steps):
    for t in range(1, steps + 1):
        J_t = algo.step(t)
        J_t = [str(i) for i in J_t]
        print(str(t) + "," + ",".join(J_t), flush=True)
