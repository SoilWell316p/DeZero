import numpy as np
import matplotlib.pyplot as plt
from DeZero import Variable
import DeZero.functions as F


def main():
    # dataset
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # initialization of weights
    I, H, O = 1, 10, 1
    W1 = Variable(0.01 * np.random.randn(I, H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.randn(H, O))
    b2 = Variable(np.zeros(O))

    # prediction of neural net
    def predict(x):
        y = F.linear(x, W1, b1)
        y = F.sigmoid_simple(y)
        # y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        return y

    lr = 0.2
    iters = 10000

    # learning
    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data
        if i % 1000 == 0:
            print(loss)

    t = np.linspace(0.0, 1.0, 100)
    plt.plot(x.T[0], y.T[0], 'bo', label="Target dots", linewidth=None)
    plt.plot(t, predict(t.reshape(100, 1)).T.data[0], 'r', label="Predicted curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
