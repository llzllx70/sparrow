
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

from Linear import *

# Create random input and output data
x1 = np.linspace(-math.pi, math.pi, 2000)
const_1 = np.linspace(1, 1, 2000)
x = Tensor(np.array([const_1, x1, x1*x1, x1*x1*x1]).T, requires_grad=False)

linear = Linear()

y1 = np.sin(x1)
y2 = np.array(y1).reshape(2000, 1)
y = Tensor(y2)

learning_rate = 3e-6
for t in range(2000):
    y_pred = linear.apply(x)
    square_ = (y_pred - y) * (y_pred - y)

    square_.backward()

    print(square_.data.sum())

    linear.update_grad(learning_rate)

y_pred = linear.forward()
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x1.data, y_pred.data)
plt.plot(x1.data, y.data)
plt.show()

