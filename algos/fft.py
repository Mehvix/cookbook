from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


def roots_of_unities(n: int) -> Union[list, np.ndarray]:
    return [np.exp(2j * np.pi * i / n) for i in range(n)]


def get_dft_matrix(n: int) -> Union[list, np.ndarray]:
    # return np.power(np.fft.fft(np.eye(n)), -1)
    return np.array([[np.exp(2j * np.pi * i * k / n) for k in range(n)] for i in range(n)])


def hyperceil(x):
    # Utility function, returns next 2^n of x
    return int(2 ** np.ceil(np.log2(x)))


def dft_naive(coeffs: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    n = len(coeffs)
    roots_needed = hyperceil(n)
    roots = roots_of_unities(roots_needed)

    return [sum([coeffs[k] * roots[k * i % roots_needed] for k in range(n)]) for i in range(n)]


def fft(coeffs, roots):
    n = len(coeffs)
    assert n == hyperceil(n)
    if n == 1: return coeffs
    even = fft(coeffs[::2], roots[::2])
    odd = fft(coeffs[1::2], roots[::2])
    z = list(zip(even, odd, roots))[:n//2]
    return [e + i * r * o for i in [complex(1,0),complex(-1,0)] for (e, o, r) in z]


def ifft(vals, roots):
    n = len(vals)
    assert n == hyperceil(n)
    return [i.real/n for i in fft([np.conjugate(v) for v in vals], roots)]


def pad(coeffs: Union[list, np.ndarray], to: int, char=0):
    """
    pad([2, 3, 4], 4)
    [2, 3, 4, 0]
    pad([1, 1, 4, 5, 1, 4], 8)
    [1, 1, 4, 5, 1, 4, 0, 0]
    """
    n = len(coeffs)
    assert n <= to
    return coeffs + [char] * (to - n)


def poly_multiply_naive(coeffs1, coeffs2):
    n1, n2 = len(coeffs1), len(coeffs2)
    n = n1 + n2 - 1
    prod_coeffs = [0] * n
    for deg in range(n):
        for i in range(max(0, deg + 1 - n2), min(n1, deg + 1)):
            prod_coeffs[deg] += coeffs1[i] * coeffs2[deg - i]
    return prod_coeffs


def poly_multiply(coeffs1, coeffs2, trim=True):
    # n = max(len(coeffs1), len(coeffs2))
    N = hyperceil(len(coeffs1)+len(coeffs2) + 1)
    roots = roots_of_unities(N)
    f1 = fft(pad(coeffs1, N), roots)
    f2 = fft(pad(coeffs2, N), roots)
    f3 = [i * j for i, j in zip(f1, f2)]
    out = ifft(f3, roots)[:len(coeffs1) + len(coeffs2) - 1]
    return np.trim_zeros(out, "b") if trim else out