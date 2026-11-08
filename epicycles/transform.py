import numpy as np
from collections.abc import Callable

def sorted_fourier_coeffs(K : int, N : int, f : Callable) -> list[tuple[int, complex]]: #f is one of the curves defined in curves.py
    """
    K개의 샘플링 포인트에서 f를 샘플링하여, fft를 통해 coefficients(C_-N~N)을 계산, 이후 absoulute value 기준으로 내림차순 정렬하여 반환하는 함수
    
    K's sample points are used to sample f, and the coefficients (C_-N~N) are calculated through fft, and then sorted in descending order based on absolute value and returned.
    """
    if N >= K//2:
        raise ValueError("N is too large. Must be less than K/2.")
    t = np.linspace(0, 2*np.pi, K, endpoint=False)
    arr = f(t)
    X = np.fft.fft(arr)
    idx = np.arange(-N, N+1) % K
    coeffs = X[idx] / K
    return sorted(zip(range(-N, N+1), coeffs), key=lambda p: abs(p[1]), reverse=True)