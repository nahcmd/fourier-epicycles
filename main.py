import epicycles.curves
import epicycles.transform
import epicycles.reconstruct
import epicycles.animate

import numpy as np
import matplotlib.pyplot as plt

#define hyperparameters

N = 1000
K = 10000
limit = 2

coeffs = epicycles.transform.sorted_fourier_coeffs(K, N, epicycles.curves.star)
err = epicycles.reconstruct.relative_error(coeffs, limit, N)
approx = epicycles.reconstruct.partial_sum(coeffs, limit, K)
ani = epicycles.animate.animate_curve(approx, err, K//100, 30)

plt.show()
