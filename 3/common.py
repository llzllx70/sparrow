import numpy as np


class Tensor(object):

    def __init__(self, data, requires_grad=False):
        self.grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad
        self.data = data
        self.accumulategrad = None  # 叶子节点

    def __repr__(self):  # 消除两边的尖括号
        return f"Tensor {self.data}"

    def check_requires_grad(self, other):
        if self.requires_grad is True:
            return True
        if other.requires_grad is True:
            return True
        return False

    def __add__(self, other):
        add_backward = AddBackward()

        if not self.requires_grad:
            add_backward.next_functions.append([None, 0])
        else:
            if self.grad_fn is None:
                if self.accumulategrad is None:
                    add_backward.next_functions.append([AccumulateGrad(self), np.array([1])])
                else:
                    add_backward.next_functions.append([self.accumulategrad, np.array([1])])
            else:
                add_backward.next_functions.append([self.grad_fn, np.array([1])])

        if not other.requires_grad:
            add_backward.next_functions.append([None, 0])
        else:
            if other.grad_fn is None:
                if other.accumulategrad is None:
                    add_backward.next_functions.append([AccumulateGrad(other), np.array([1])])
                else:
                    add_backward.next_functions.append([other.accumultegrad, np.array([1])])
            else:
                add_backward.next_functions.append([other.grad_fn, np.array([1])])

        requires_grad = self.check_requires_grad(other)
        v = self.data + other.data
        variable = Tensor(v, requires_grad=requires_grad)
        variable.grad_fn = add_backward
        return variable

    def __sub__(self, other):
        sub_backward = SubBackward()

        if not self.requires_grad:
            sub_backward.next_functions.append([None, 0])
        else:
            if self.grad_fn is None:
                if self.accumulategrad is None:
                    sub_backward.next_functions.append([AccumulateGrad(self), np.array([1])])
                else:
                    sub_backward.next_functions.append([self.accumulategrad, np.array([1])])
            else:
                sub_backward.next_functions.append([self.grad_fn, np.array([1])])

        if not other.requires_grad:
            sub_backward.next_functions.append([None, 0])
        else:
            if other.grad_fn is None:
                if other.accumulategrad is None:
                    sub_backward.next_functions.append([AccumulateGrad(other), np.array([-1])])
                else:
                    sub_backward.next_functions.append([other.accumultegrad, np.array([-1])])
            else:
                sub_backward.next_functions.append([other.grad_fn, np.array([-1])])

        requires_grad = self.check_requires_grad(other)
        v = self.data - other.data
        variable = Tensor(v, requires_grad=requires_grad)
        variable.grad_fn = sub_backward
        return variable

    def __mul__(self, other):
        mul_backward = MulBackward()

        if not self.requires_grad:
            mul_backward.next_functions.append([None, 0])
        else:
            if self.grad_fn is None:
                if self.accumulategrad is None:
                    mul_backward.next_functions.append([AccumulateGrad(self), other.data])
                else:
                    mul_backward.next_functions.append([self.accumulategrad, other.data])
            else:
                mul_backward.next_functions.append([self.grad_fn, other.data])

        if not other.requires_grad:
            mul_backward.next_functions.append([None, 0])
        else:
            if other.grad_fn is None:
                if other.accumulategrad is None:
                    mul_backward.next_functions.append([AccumulateGrad(other), self.data])
                else:
                    mul_backward.next_functions.append([other.accumultegrad, self.data])
            else:
                mul_backward.next_functions.append([other.grad_fn, self.data])

        requires_grad = self.check_requires_grad(other)
        v = self.data * other.data
        variable = Tensor(v, requires_grad=requires_grad)
        variable.grad_fn = mul_backward
        return variable

    def backward(self):
        if self.grad_fn is not None:
            self.grad_fn.run(1)


class AccumulateGrad(object):

    def __init__(self, variable):
        self.variable = variable

    def run(self, input_):
        if self.variable.grad is None:
            self.variable.grad = input_.sum()  # 引入多维
        else:
            self.variable.grad += input_.sum()


class AddBackward(object):

    def __init__(self):
        self.next_functions = []

    def run(self, input_):
        for f in self.next_functions:
            if f[0] is not None:
                f[0].run(f[1] * input_)


class SubBackward(object):

    def __init__(self):
        self.next_functions = []

    def run(self, input_):
        for f in self.next_functions:
            if f[0] is not None:
                f[0].run(f[1] * input_)


class MulBackward(object):

    def __init__(self):
        self.next_functions = []

    def run(self, input_):
        for f in self.next_functions:
            if f[0] is not None:
                f[0].run(f[1] * input_)


