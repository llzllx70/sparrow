from common import *

x = Tensor(3)
b = Tensor(2)


def _add():
    y = add_t_i(x, 3)
    y2 = add_t_i(y, 3)
    y2.backward()


def _mul():
    y = mul_t_i(x, 3)
    y2 = mul_t_i(y, 4)
    y3 = mul_t_i(y2, 5)
    y3.backward()


def _mix():
    y = mul_t_t(x, x)
    y1 = mul_t_t(y, x)
    y2 = mul_t_i(y1, 3)
    y2.backward()


def _mix_2():
    """
    3 * x ** 2
    """
    z1 = mul_t_i(x, 2)
    z2 = mul_t_i(x, 3)
    z = add_t_t(z1, z2)
    z.backward()
    print(z)


def sub():
    z1 = sub_t_t(x, b)
    z = sub_t_t(z1, b)
    z.backward()
    print(z)


sub()
print('end')
