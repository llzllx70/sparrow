
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

from common import *

# Create random input and output data
x1 = np.linspace(-math.pi, math.pi, 2000)
# x = np.linspace(2, 2, 10)
y1 = np.sin(x1)

x = Tensor(x1)
y = Tensor(y1)

# y = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# Randomly initialize weights
a = Tensor(np.random.randn(), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)
c = Tensor(np.random.randn(), requires_grad=True)
d = Tensor(np.random.randn(), requires_grad=True)

# a = Tensor(1)
# b = Tensor(1)
# c = Tensor(1)
# d = Tensor(1)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x * x + d * x * x * x
    square_ = (y_pred - y) * (y_pred - y)

    square_.backward()

    print(square_.data.sum())

    # Update weights
    a.data -= learning_rate * a.grad
    b.data -= learning_rate * b.grad
    c.data -= learning_rate * c.grad
    d.data -= learning_rate * d.grad

    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

y_pred = a.data + b.data * x.data + c.data * x.data ** 2 + d.data * x.data ** 3
print(f'Result: y = {a.data} + {b.data}x + {c.data}x^2 + {d.data}x^3')
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x.data, y_pred)
plt.plot(x.data, y.data)
plt.show()

