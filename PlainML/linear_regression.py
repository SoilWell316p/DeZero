import argparse


import numpy as np
from DeZero import Variable
import DeZero.functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    # toy dataset
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = F.matmul(x, W) + b
        return y

    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return F.sum(diff ** 2) / len(diff)

    for i in range(args.iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= args.lr * W.grad.data
        b.data -= args.lr * b.grad.data
        print(W, b, loss)


if __name__ == "__main__":
    main()
