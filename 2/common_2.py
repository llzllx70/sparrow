
class Tensor(object):

    def __init__(self, value):
        self.grad = None
        self.grad_fn = None
        self.requires_grad = True
        self.value = value
        self.accumulategrad = None  # 叶子节点

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


def add_t_i(k, m):
    add_backward = AddBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            add_backward.next_functions.append([AccumulateGrad(k), 1])
        else:
            add_backward.next_functions.append([k.accumultegrad, 1])
    else:
        add_backward.next_functions.append([k.grad_fn, 1])

    if type(m) is not Tensor:
        add_backward.next_functions.append([None, 0])

    variable = Tensor(k.value + m)
    variable.grad_fn = add_backward

    return variable


def add_t_t(k, m):
    add_backward = AddBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            add_backward.next_functions.append([AccumulateGrad(k), 1])
        else:
            add_backward.next_functions.append([k.accumultegrad, 1])
    else:
        add_backward.next_functions.append([k.grad_fn, 1])

    if m.grad_fn is None:
        if m.accumulategrad is None:
            add_backward.next_functions.append([AccumulateGrad(m), 1])
        else:
            add_backward.next_functions.append([m.accumultegrad, 1])
    else:
        add_backward.next_functions.append([m.grad_fn, 1])

    variable = Tensor(k.value + m.value)
    variable.grad_fn = add_backward

    return variable


def sub_t_n(k, m):
    add_backward = SubBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            add_backward.next_functions.append([AccumulateGrad(k), 1])
        else:
            add_backward.next_functions.append([k.accumultegrad, 1])
    else:
        add_backward.next_functions.append([k.grad_fn, 1])

    if type(m) is not Tensor:
        add_backward.next_functions.append([None, 0])

    variable = Tensor(k.value - m)
    variable.grad_fn = add_backward

    return variable


def sub_t_t(k, m):
    sub_backward = SubBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            sub_backward.next_functions.append([AccumulateGrad(k), 1])
        else:
            sub_backward.next_functions.append([k.accumultegrad, 1])
    else:
        sub_backward.next_functions.append([k.grad_fn, 1])

    if m.grad_fn is None:
        if m.accumulategrad is None:
            sub_backward.next_functions.append([AccumulateGrad(m), -1])
        else:
            sub_backward.next_functions.append([m.accumultegrad, -1])
    else:
        sub_backward.next_functions.append([m.grad_fn, -1])

    variable = Tensor(k.value - m.value)
    variable.grad_fn = sub_backward

    return variable


def mul_t_i(k, m):
    mul_backward = MulBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            mul_backward.next_functions.append([AccumulateGrad(k), m])
        else:
            mul_backward.next_functions.append([k.accumultegrad, m])
    else:
        mul_backward.next_functions.append([k.grad_fn, m])

    if type(m) is not Tensor:
        mul_backward.next_functions.append([None, 0])

    variable = Tensor(k.value * m)
    variable.grad_fn = mul_backward

    return variable


def mul_t_n(k, m):
    """
    :param k: tensor
    :param m: numpy
    :return:
    """
    mul_backward = MulBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            mul_backward.next_functions.append([AccumulateGrad(k), m])
        else:
            mul_backward.next_functions.append([k.accumultegrad, m])
    else:
        mul_backward.next_functions.append([k.grad_fn, m])

    if type(m) is not Tensor:
        mul_backward.next_functions.append([None, 0])

    variable = Tensor(k.value * m)
    variable.grad_fn = mul_backward

    return variable


def mul_t_t(k, m):
    mul_backward = MulBackward()

    if k.grad_fn is None:
        if k.accumulategrad is None:
            mul_backward.next_functions.append([AccumulateGrad(k), m.value])
        else:
            mul_backward.next_functions.append([k.accumultegrad, m.value])
    else:
        mul_backward.next_functions.append([k.grad_fn, m.value])

    if m.grad_fn is None:
        if m.accumulategrad is None:
            mul_backward.next_functions.append([AccumulateGrad(m), k.value])
        else:
            mul_backward.next_functions.append([m.accumultegrad, k.value])
    else:
        mul_backward.next_functions.append([m.grad_fn, k.value])

    variable = Tensor(k.value * m.value)
    variable.grad_fn = mul_backward

    return variable


"""
思路:
1. 在运算的过程中生成反向函数
2. 运算产生的结果变量的梯度指向该反向函数
"""

