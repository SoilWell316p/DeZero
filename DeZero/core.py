class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator    # get function
        if f in not None:
            x = f.input    # get the input of the function
            x.grad = f.backward(self.grad)    # call backward method of the function
            x.backward()



class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)    # forward method implements specific computation
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()




