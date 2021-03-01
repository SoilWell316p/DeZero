import numpy as np
import DeZero
from DeZero import cuda, utils
from DeZero.core import Function, Variable, as_variable, as_array


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

