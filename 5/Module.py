
from common_5 import *


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


class Module:

    def __init__(self):
        self.input = None

    def forward(self, xx):
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


class Linear:

    """
    Linear 属于 Module, 只是将weight和Function封装起来，并没有任何的特殊之处
    需要特别考虑的是jacobian matrix的求导操作

    现在需要考虑对于正常的矩阵运算怎么进行, 可以先不考虑Module
    其中一个笨办法就是将weight的每个分量都设计为一个变量
    """

    def __init__(self):
        # Tensor
        self.weight1 = Tensor(np.random.rand(1), requires_grad=True)
        self.weight2 = Tensor(np.random.rand(1), requires_grad=True)
        self.weight3 = Tensor(np.random.rand(1), requires_grad=True)

        self.bias = Tensor(np.random.rand(1), requires_grad=True)

    def forward(self, x):
        return self.weight1 * x + self.weight2 * x * x + self.weight3 * x * x * x + self.bias

    def update_grad(self, learning_rate):
        self.weight1.data -= learning_rate * self.weight1.grad
        self.weight2.data -= learning_rate * self.weight2.grad
        self.weight3.data -= learning_rate * self.weight3.grad
        self.bias.data -= learning_rate * self.bias.grad

        self.weight1.grad = None
        self.weight2.grad = None
        self.weight3.grad = None
        self.bias.grad = None


