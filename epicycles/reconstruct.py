import numpy as np

def relative_error(sorted_fourier_coeffs, limit, N):
    numerator = 0
    demominator = 0
    for i in range(2*N+1):
        if i < limit:
            numerator += abs(sorted_fourier_coeffs[i][1])**2
        demominator += abs(sorted_fourier_coeffs[i][1])**2
    return 1-np.sqrt(numerator/demominator)

def partial_sum(sorted_fourier_coeffs, limit, K):
    X = np.zeros(K, dtype=complex)
    for i in range(limit):
        n, c = sorted_fourier_coeffs[i]    
        X[n % K] += c                      
    return np.fft.ifft(X) * K