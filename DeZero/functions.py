import numpy as np
from DeZero.core import Function, Variable, as_array
import math
# from DeZero import cuda, utils
# from DeZero.core import Function, Variable, as_variable, as_array


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    f = Exp()
    return f(x)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


# テイラー展開で近似してみる
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i /math.factorial(2 * i + 1)
        t = c * x **(2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

