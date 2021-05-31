
from common_4 import *


class F3Backward(object):

    def __init__(self):
        self.next_functions = []

    def run(self, input_):
        for f in self.next_functions:
            if f[0] is not None:
                f[0].run(f[1] * input_)


class AutogradBase:

    def __init__(self):
        self.input = None

    def forward(self):
        pass

    def backward(self):
        pass

    def apply(self, input_):
        self.input = input_

        v = self.forward()

        f3_backward = F3Backward()
        f3_backward.next_functions.append([self.input.grad_fn, self.backward()])

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


class LegendrePolynomial3(AutogradBase):

    def forward(self):
        return 0.5 * (5 * self.input.data ** 3 - 3 * self.input.data)

    def backward(self):
        return np.array([1.5]) * (5 * self.input.data ** 2 - 1)

