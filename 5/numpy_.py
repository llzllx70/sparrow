
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

from Module import *

# Create random input and output data
x1 = np.linspace(-math.pi, math.pi, 2000)
x = Tensor(x1)
# xx = Tensor(np.array([x1, x1*x1, x1*x1*x1]).T, requires_grad=False)

linear = Linear()

y1 = np.sin(x1)
y = Tensor(y1)

learning_rate = 1e-6
for t in range(2000):
    y_pred = linear.forward(x)
    square_ = (y_pred - y) * (y_pred - y)

    square_.backward()

    print(square_.data.sum())

    linear.update_grad(learning_rate)

y_pred = linear.forward(x)
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x1.data, y_pred.data)
plt.plot(x1.data, y.data)
plt.show()

