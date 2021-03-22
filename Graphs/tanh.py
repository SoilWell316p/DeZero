import argparse

import numpy as np
from DeZero import Variable
from DeZero.utils import plot_dot_graph
import DeZero.functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--out", type=str, default="tanh.png")
    args = parser.parse_args()

    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)

    iters = args.iters

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = 'gx' + str(iters+1)
    plot_dot_graph(gx, verbose=args.verbose, to_file=args.out)


if __name__ == "__main__":
    main()