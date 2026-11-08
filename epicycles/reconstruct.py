import numpy as np

def relative_error(sorted_fourier_coeffs : list[tuple[int, complex]], limit : int, N : int) -> float:
    """
    transforom에서 정의된 sorted_fourier_coeffs를 이용하여, limit개의 coefficient만을 사용했을 때의 상대오차를 계산하는 함수
    relative error is calculated using sorted_fourier_coeffs defined in transform, when only limit coefficients are used.
    """
    numerator = 0
    demominator = 0
    for i in range(2*N+1):
        if i < limit:
            numerator += abs(sorted_fourier_coeffs[i][1])**2
        demominator += abs(sorted_fourier_coeffs[i][1])**2
    return np.sqrt(1 - numerator/demominator)

def partial_sum(sorted_fourier_coeffs : list[tuple[int, complex]], limit : int, K : int) -> np.ndarray:
    """
    transforom에서 정의된 sorted_fourier_coeffs를 이용하여, limit개의 coefficient만을 사용했을 때의 partial sum을 계산하는 함수
    partial sum is calculated using sorted_fourier_coeffs defined in transform, when only limit coefficients are used.
    """
    X = np.zeros(K, dtype=complex)
    for i in range(limit):
        n, c = sorted_fourier_coeffs[i]    
        X[n % K] += c                      
    return np.fft.ifft(X) * K