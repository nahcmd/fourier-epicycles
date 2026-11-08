import epicycles.curves
import epicycles.transform
import epicycles.reconstruct
import epicycles.animate

import numpy as np
import matplotlib.pyplot as plt

#define hyperparameters

N = 1000
K = 10000
limit = 3

sorted_fourier_coeffs = epicycles.transform.sorted_fourier_coeffs(K, N, epicycles.curves.star)

relative_error = epicycles.reconstruct.relative_error(sorted_fourier_coeffs, limit, N)
partial_sum = epicycles.reconstruct.partial_sum(sorted_fourier_coeffs, limit, K)

animate = epicycles.animate.animate_curve(partial_sum, relative_error, K//100, 30)

plt.show()
