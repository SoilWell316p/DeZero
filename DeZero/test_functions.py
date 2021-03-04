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


def numerical_diff(x, f, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

