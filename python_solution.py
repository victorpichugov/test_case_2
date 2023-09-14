import math
import numpy as np
from matplotlib import pyplot as plt


def dot(x, y):
    return sum([x * y for x, y in zip(x, y)])


def exact_solution(t):
    return -math.e ** (-3 * t) / 12 * t * (-36 - 54 * t + 16 * t ** 2 + 129 * t ** 3)


def diff(x, y):
    return [x - y for x, y in zip(x, y)]


def sums(x, y):
    return [x + y for x, y in zip(x, y)]


def product(vec, scal):
    return [x * scal for x in vec]


def column(matrix, i):
    return [row[i] for row in matrix]


class Solution:

    def euler_method(self, m: int, left: float, right: float, coeff: list[float], initial_conditions: list[float]):
        n = len(initial_conditions)  # порядок дифф. уравнения
        h = (right - left) / m  # шаг сетки
        A = [[0] * m for j in range(n)]  # создали матрицу n * m

        # Реализация начальных условий
        for i in range(n):
            A[i][0] = initial_conditions[i]

        for i in range(1, m):
            for j in range(n):
                if j != n - 1:
                    A[j][i] = A[j][i - 1] + h * A[j + 1][i - 1]
                else:
                    A[j][i] = A[j][i - 1] - h * dot(coeff[1:], column(A, i - 1)[::-1]) / coeff[0]

        return A[0]

    def runge_kutta(self, m: int, left: float, right: float, coeff: list[float], initial_conditions: list[float]):
        n = len(initial_conditions)  # порядок дифф. уравнения
        h = (right - left) / m  # шаг сетки
        A = [[0] * m for j in range(n)]  # создали матрицу n * m
        B = [[[0] * 4 for i in range(n)] for k in range(m)]  # матрица коэффициентов K, L, M, N, P
        # Реализация начальных условий
        for i in range(n):
            A[i][0] = initial_conditions[i]

        for k in range(m - 1):
            for i in range(4):
                for j in range(n):
                    if i == 0:
                        if j != n - 1:
                            B[k][j][i] = h * A[j + 1][k]
                        else:
                            B[k][j][i] = -h * dot(coeff[1:], column(A, k)[::-1]) / coeff[0]
                    if i == 1 or i == 2:
                        if j != n - 1:
                            B[k][j][i] = h * (A[j + 1][k] + B[k][j + 1][i - 1] / 2)
                        else:
                            B[k][j][i] = -h * dot(coeff[1:], sums(column(A, k), product(column(B[k], i - 1), 0.5))[::-1]) / coeff[
                                             0]
                    if i == 3:
                        if j != n - 1:
                            B[k][j][i] = h * (A[j + 1][k] + B[k][j + 1][i - 1])
                        else:
                            B[k][j][i] = -h * dot(coeff[1:], sums(column(A, k), column(B[k], i - 1))[::-1]) / coeff[0]

            # Обновление значений
            for j in range(n):
                A[j][k + 1] = A[j][k] + (B[k][j][0] + 2 * B[k][j][1] + 2 * B[k][j][2] + B[k][j][3]) / 6

        return A[0]


if __name__ == '__main__':
    M = 5
    a, b = 0, 5
    x = np.linspace(a, b, M)
    coeff = [1, 15, 90, 270, 405, 243]
    initial_conditions = [0, 3, -9, -8, 0]

    N = range(20, 60, 5)
    x_1 = Solution()
    x_2 = Solution()

    max_diff_euler = [max([abs(el) for el in diff(exact_solution(np.linspace(a, b, el)), x_1.euler_method(el, a, b, coeff, initial_conditions))]) for el in N]
    max_diff_r_g = [max([abs(el) for el in diff(exact_solution(np.linspace(a, b, el)), x_2.runge_kutta(el, a, b, coeff, initial_conditions))]) for el in N]
    plt.plot(N, max_diff_euler, label="error_euler")
    plt.plot(N, max_diff_r_g, label="error_runge_kutta")
    plt.scatter(N, max_diff_euler, marker='o')
    plt.scatter(N, max_diff_r_g, marker='^')
    plt.grid()
    plt.legend()
    plt.xlabel('N - число узлов')
    plt.ylabel('Максимальная разница')
    plt.show()
