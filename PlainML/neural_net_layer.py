import numpy as np
from DeZero import Variable
import DeZero.functions as F
import DeZero.layers as L


def main():
    # dataset
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    l1 = L.Linear(10)
    l2 = L.Linear(1)

    def predict(x):
        y = l1(x)
        y = F.sigmoid_simple(y)
        y = l2(y)
        return y

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)


if __name__ == "__main__":
    main()
