import numpy as np
import matplotlib.pyplot as plt
from DeZero import Variable
from DeZero.functions import sin


def main():
    x = Variable(np.linspace(-7, 7, 100))
    y = sin(x)
    y.backward(create_graph=True)

    logs = [y.data.flatten()]

    # 3階微分
    for i in range(3):
        logs.append(x.grad.data.flatten())
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, log in enumerate(logs):
        plt.plot(x.data, log, label=labels[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
