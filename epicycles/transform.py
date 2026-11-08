import numpy as np
from collections.abc import Callable

def sorted_fourier_coeffs(K : int, N : int, f : Callable) -> list[tuple[int, complex]]:
    if N >= K//2:
        raise ValueError("N is too large. Must be less than K/2.")
    t = np.linspace(0, 2*np.pi, K, endpoint=False)
    arr = f(t)
    X = np.fft.fft(arr)
    idx = np.arange(-N, N+1) % K
    coeffs = X[idx] / K
    return sorted(zip(range(-N, N+1), coeffs), key=lambda p: abs(p[1]), reverse=True)