
from common_6 import *


class F3Backward(object):

    def __init__(self):
        self.next_functions = []

    def run(self, input_):
        for f in self.next_functions:
            if f[0] is not None:
                f[0].run(np.matmul(f[1], input_))  # [2000, 4] * [2000, 1]


class AutogradBase:

    def __init__(self):
        self.input = None

    def forward(self):
        pass

    def backward(self):
        pass

    def update_grad(self, learning_rate):
        pass

    def apply(self, input_):
        self.input = input_

        v = self.forward()

        f3_backward = F3Backward()
        f3_backward.next_functions.append([self.weight.grad_fn, self.input.data.T])

        variable = Tensor(v, requires_grad=True)
        variable.grad_fn = f3_backward

        return variable


class Square(AutogradBase):

    def forward(self):
        return self.input.data * self.input.data

    def backward(self):
        return np.array([2]) * self.input.data


class Cube(AutogradBase):

    def forward(self):
        return self.input.data * self.input.data * self.input.data

    def backward(self):
        return np.array([3]) * self.input.data * self.input.data


class Linear(AutogradBase):

    def __init__(self):
        super(Linear).__init__()
        self.weight = Tensor(np.random.rand(4, 1), requires_grad=True)

    def forward(self):
        return np.matmul(self.input.data, self.weight.data)

    def backward(self):
        return np.array([1])

    def update_grad(self, learning_rate):
        self.weight.data -= learning_rate * self.weight.grad
        self.weight.grad = None


class MM(AutogradBase):

    def forward(self):
        return self.input.data * self.input.data * self.input.data

    def backward(self):
        return np.array([3]) * self.input.data * self.input.data


class LegendrePolynomial3(AutogradBase):

    def forward(self):
        return 0.5 * (5 * self.input.data ** 3 - 3 * self.input.data)

    def backward(self):
        return np.array([1.5]) * (5 * self.input.data ** 2 - 1)


