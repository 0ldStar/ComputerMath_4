import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

BEGIN, END, INC = 0, 5, 0.001
x = np.arange(BEGIN, END, INC)

POINTS = [0, 1, 3, 5]


def func(_x: float) -> float:
    return np.cos(_x + 0.3)


def der1F(_x: float) -> float:
    return -np.sin(_x + 0.3)


def der2F(_x: float) -> float:
    return -np.cos(_x + 0.3)


def der3F(_x: float) -> float:
    return np.sin(_x + 0.3)


def der4F(_x: float) -> float:
    return np.cos(_x + 0.3)


def P1X(_x: float, a: float) -> float:
    return func(a) + der1F(a) * (_x - a)


def P2X(_x: float, a: float) -> float:
    return P1X(_x, a) + (der2F(a) * (_x - a) ** 2) / 2


def P3X(_x: float, a: float) -> float:
    return P2X(_x, a) + (der3F(a) * (_x - a) ** 3) / 6


def P4X(_x: float, a: float) -> float:
    return P3X(_x, a) + (der4F(a) * (_x - a) ** 4) / 24


def err1x(_x: float, a: float) -> float:
    return abs(func(_x) - P1X(_x, a))


def err2x(_x: float, a: float) -> float:
    return abs(func(_x) - P2X(_x, a))


def err3x(_x: float, a: float) -> float:
    return abs(func(_x) - P3X(_x, a))


def err4x(_x: float, a: float) -> float:
    return abs(func(_x) - P4X(_x, a))


def find_mistakes(point: float):
    mis1, mis2, mis3, mis4 = [], [], [], []

    j = BEGIN
    while j < END:
        mis1.append(err1x(j, point))
        mis2.append(err2x(j, point))
        mis3.append(err3x(j, point))
        mis4.append(err4x(j, point))
        j += INC
    return mis1, mis2, mis3, mis4


def draw_polinomial(point):
    f1, f2, f3, f4 = [], [], [], []

    j = BEGIN
    while j < END:
        f1.append(P1X(j, point))
        f2.append(P2X(j, point))
        f3.append(P3X(j, point))
        f4.append(P4X(j, point))
        j += INC
    plt.figure(figsize=(10, 10))
    plt.title("polynomial x=" + str(point))
    plt.plot(x, func(x), label='original function', color='blue')
    plt.plot(x, f1, label='polynomial of degree 1', color='red')
    plt.plot(x, f2, label='polynomial of degree 2', color='yellow')
    plt.plot(x, f3, label='polynomial of degree 3', color='green')
    plt.plot(x, f4, label='polynomial of degree 4', color='magenta')
    plt.grid()
    plt.legend()
    plt.show()


def draw_mistakes(mis1: list, mis2: list, mis3: list, mis4: list, point: int):
    plt.figure(figsize=(10, 10))
    plt.title("polynomial error x=" + str(point))
    plt.plot(x, mis1, label='polynomial error of degree 1', color='red')
    plt.plot(x, mis2, label='polynomial error of degree 2', color='yellow')
    plt.plot(x, mis3, label='polynomial error of degree 3', color='green')
    plt.plot(x, mis4, label='polynomial error of degree 4', color='magenta')
    plt.grid()
    plt.legend()
    plt.show()


def newton_poly(x, newP, newC):
    b0, b1, b2, b3 = newC[0], newC[1], newC[2], newC[3]
    x0, x1, x2 = newP[0], newP[1], newP[2]
    return b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1) + b3 * (x - x0) * (x - x1) * (x - x2)


def draw_newton_poly(newP, newC):
    plt.figure(figsize=(10, 10))
    plt.title("function graph and newton polynomial")
    plt.plot(x, func(x), color='black')
    plt.plot(x, newton_poly(x, newP, newC))
    for i in range(len(POINTS)):
        plt.scatter(POINTS[i], func(POINTS[i]))
    plt.grid()
    plt.show()


def sliding_filling(length):
    points = []
    b0, b1, b2 = [], [], []
    for i in range(length + 1):
        points.append(BEGIN + 5 / length * i)

    for i in range(len(points) - 2):
        b0.append(func(points[i]))

        b1.append((func(points[i + 1]) - func(points[i])) / (points[i + 1] - points[i]))

        tc = (func(points[i + 1]) - func(points[i])) / (points[i + 1] - points[i])
        tmp = func(points[i + 2]) - func(points[i]) - tc * (points[i + 2] - points[i])
        b2.append(tmp / ((points[i + 2] - points[i]) * (points[i + 2] - points[i + 1])))

    return points, b0, b1, b2


def sPol(b0, b1, b2, x0, x1, x):
    return b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1)


def draw_sliding_poly(points, b0, b1, b2):
    plt.title("sliding polynomials (n = " + str(len(points) - 2) + ")")
    for i in range(len(points) - 2):
        xs = np.arange(points[i], points[i + 2], INC)
        plt.plot(xs, sPol(b0[i], b1[i], b2[i], points[i], points[i + 1], xs))
    for i in range(len(points)):
        plt.scatter(points[i], func(points[i]))
    plt.grid()
    plt.show()


# for i in range(len(POINTS)):
#     mis1, mis2, mis3, mis4 = find_mistakes(POINTS[i])
#     draw_mistakes(mis1, mis2, mis3, mis4, POINTS[i])
# print(POINTS[i])
# print(mis1[POINTS[i]], mis2[POINTS[i]], mis3[POINTS[i]], mis4[POINTS[i]])

#
# points = [0, 1, 3, 5]
# coefficients = [0.955, -0.687, -0.150, 0.352]
# draw_newton_poly(points, coefficients)
#
# for i in range(3, 12, 1):
#     slidingPoints, b0, b1, b2 = sliding_filling(i)
#     draw_sliding_poly(slidingPoints, b0, b1, b2)
#
# plt.plot(x, func(x))
# plt.show()
# for i in range(len(POINTS)):
#     draw_polinomial(POINTS[i])

# def printErr(point):
#     print(point)
#     print(err1x(point, 0))
#     print(err2x(point, 1))
#     print(err3x(point, 3))
#     print(err4x(point, 5))

# def printErr(point):
#     print(point)
#     print(integrate.quad(err1x, 0, 5, args=(point))[0])
#     print(integrate.quad(err2x, 0, 5, args=(point))[0])
#     print(integrate.quad(err3x, 0, 5, args=(point))[0])
#     print(integrate.quad(err4x, 0, 5, args=(point))[0])
#
#
# printErr(0)
# printErr(2.5)
# printErr(5)
