
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from common_2 import *

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
# x = np.linspace(2, 2, 10)
y = np.sin(x)

# y = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# Randomly initialize weights
a = Tensor(np.random.randn())
b = Tensor(np.random.randn())
c = Tensor(np.random.randn())
d = Tensor(np.random.randn())

# a = Tensor(1)
# b = Tensor(1)
# c = Tensor(1)
# d = Tensor(1)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    # y_pred = a + b * x + c * x ** 2 + d * x ** 3

    bx = mul_t_n(b, x)

    x2 = x * x
    cx2 = mul_t_n(c, x2)

    x3 = x * x * x
    dx3 = mul_t_n(d, x3)

    a_bx = add_t_t(a, bx)
    a_bx_cx2 = add_t_t(a_bx, cx2)
    y_pred = add_t_t(a_bx_cx2, dx3)
    print('dd')

    sub_ = sub_t_n(y_pred, y)
    square_ = mul_t_t(sub_, sub_)

    square_.backward()

    print(square_.value.sum())

    # Update weights
    a.value -= learning_rate * a.grad
    b.value -= learning_rate * b.grad
    c.value -= learning_rate * c.grad
    d.value -= learning_rate * d.grad

    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

y_pred = a.value + b.value * x + c.value * x ** 2 + d.value * x ** 3
print(f'Result: y = {a.value} + {b.value}x + {c.value}x^2 + {d.value}x^3')
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y_pred)
plt.plot(x, y)
plt.show()
