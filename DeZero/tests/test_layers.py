import unittest
import numpy as np
from DeZero.core import Variable, Parameter
from DeZero.layers import Layer


class TestPlainLayers(unittest.TestCase):
    def test_simple_layer(self):
        layer = Layer()

        layer.p1 = Parameter(np.array(1.0))
        layer.p2 = Parameter(np.array(2.0))
        layer.p3 = Variable(np.array(3.0))
        layer.p4 = 'test'

        self.assertEqual(layer._params, {'p2', 'p1'})


