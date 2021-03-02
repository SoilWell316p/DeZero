class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)    # forward method implements specific computation
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()




