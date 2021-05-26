import numpy as np
from common_2 import Tensor, mul_t_n

b = Tensor(2)
x = np.asarray([1, 2, 3])


def _mul():
    y = mul_t_n(b, x)
    y.backward()


_mul()
print('end')
