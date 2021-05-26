
from common_4 import *


a = Tensor(np.array([2]), requires_grad=True)
b = Tensor(np.array([3]), requires_grad=True)

c = a * b
# y = Square().apply(c)   # (a+b)****4 = 4 * (a+b) *** 3 = 4 * 5 *** 3
# y.backward()
# a.grad = None
# b.grad = None

# y2 = Cube().apply(c)   # (a+b)****4 = 4 * (a+b) *** 3 = 4 * 5 *** 3
# y2.backward()

y = LegendrePolynomial3().apply(a)
y.backward()

pass

# 3 * a**2 * b *** 3 = 3 * 8 * 27
