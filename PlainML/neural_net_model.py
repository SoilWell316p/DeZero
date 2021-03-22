import numpy as np
from DeZero import Variable, Model
import DeZero.layers as L
import DeZero.functions as F


def main():
    # dataset
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # hyper parameters
    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    # model definition
    class TwoLayerNet(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(out_size)

        def forward(self, x):
            y = F.sigmoid_simple(self.l1(x))
            y = self.l2(y)
            return y

    model = TwoLayerNet(hidden_size, 1)

    # learning initiation
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss)


if __name__ == "__main__":
    main()
