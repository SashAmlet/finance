import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

d = 3
r = 0.0237

sigma = np.array([[0.000961, -0.000798, 0.00036],
                  [-0.000798, 0.01097, -0.001418],
                  [0.00036, -0.001418, 0.003653]])

mu = np.array([0.0302, 0.037, 0.0237])

"""Формування матриці A та вектора b"""

A = np.zeros((d + 2, d + 2))
A[:d, :d] = 2 * sigma
A[:d, d] = -mu
A[:d, d + 1] = 1
A[d, :d] = mu
A[d + 1, :d] = 1

# for i in range(0, d + 2):
#     A[i, 1] = 0
#     A[1, i] = 0

# A[1,1] = 1

b = np.zeros(d + 2)
b[d] = r
b[d + 1] = 1

print(A)
print(b)

# """Обчислення та висновки"""

# Регуляризація Тихонова
lambda_reg = 1e-5  # Параметр регуляризації
A_reg = A + lambda_reg * np.eye(A.shape[0])

# Обчислення оберненої матриці з регуляризацією
A_inv_reg = np.linalg.inv(A_reg)

# Множення оберненої матриці на вектор b
result = A_inv_reg @ b

print(result)