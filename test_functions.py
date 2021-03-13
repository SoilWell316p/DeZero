import numpy as np
import unittest

from DeZero.core import *
from DeZero.functions import *


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(x, square)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class AddTest(unittest.TestCase):
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(y.data, 6.0)
        self.assertEqual(x.grad, 2.0)

    def test_repetitive_backward(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, 2.0)
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, 3.0)


class MulTest(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        y = a * b
        self.assertEqual(y.data, 6.0)


class CombinedFuncsTest(unittest.TestCase):
    def test_complicated_backward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(y.data, 32.0)
        self.assertEqual(x.grad, 64.0)


class ComplexFuncTest(unittest.TestCase):
    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual((x.grad, x.grad), (2.0, 2.0))

    def test_rosenbrock(self):
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))
        y = rosenbrock(x0, x1)
        y.backward()
        self.assertEqual(y.data, 0.0)


class OptimizationTest(unittest.TestCase):
    def test_optim_rosenbrock(self):
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))
        lr = 0.001
        iters = 50000

        for i in range(iters):
            y = rosenbrock(x0, x1)

            x0.cleargrad()
            x1.cleargrad()
            y.backward()

            x0.data = x0.data - lr * x0.grad
            x1.data = x1.data - lr * x1.grad

        # 50000回回すと、誤差は10の-8乗以下になる
        self.assertEqual((round(x0.data, 8), round(x1.data, 8)), (1.0, 1.0))

    def test_second_order_differentiation(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y

        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, 24.0)

        gx = x.grad
        # 勾配が加算されないよう、リセットする
        x.cleargrad()
        gx.backward()
        self.assertEqual(x.grad.data, 44.0)

    def test_newton_method(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y

        x = Variable(np.array(2.0))
        iters = 10

        for i in range(iters):
            y = f(x)
            x.cleargrad()
            y.backward(create_graph=True)

            gx = x.grad
            x.cleargrad()
            gx.backward()
            gx2 = x.grad

            x.data = x.data - gx.data / gx2.data

        self.assertEqual(x.data, 1.0)


def numerical_diff(x, f, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y
