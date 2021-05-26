
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

from common_4 import *

# Create random input and output data
x1 = np.linspace(-math.pi, math.pi, 2000)
# x = np.linspace(2, 2, 10)
y1 = np.sin(x1)

x = Tensor(x1)
y = Tensor(y1)

# y = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# Randomly initialize weights
a = Tensor(np.array([0.0]), requires_grad=True)
b = Tensor(np.array([-1.0]), requires_grad=True)
c = Tensor(np.array([0.0]), requires_grad=True)
d = Tensor(np.array([0.3]), requires_grad=True)


learning_rate = 5e-6
for t in range(2000):
    y_pred = a + b * LegendrePolynomial3().apply(c + d * x)
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


y_pred = a + b * LegendrePolynomial3().apply(c + d * x)
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x.data, y_pred.data)
plt.plot(x.data, y.data)
plt.show()

