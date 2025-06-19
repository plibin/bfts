def print_header(m, out):
    header = ['m %i' % i for i in range(1, m + 1)]
    out.write("t," + ",".join(header) + "\n")
    out.flush()
    
def run(algo, steps, out):
    for t in range(1, steps + 1):
        J_t = algo.step(t)
        J_t = [str(i) for i in J_t]
        out.write(str(t) + "," + ",".join(J_t) + "\n")
        out.flush()
