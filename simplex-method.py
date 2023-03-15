import numpy as np
from scipy import optimize
import math


# чтобы красиво печатало матрицы
np.set_printoptions(linewidth=10000)
np.set_printoptions(precision=2, floatmode='fixed')

# вводные значения
A = np.array([[10, 35, 2,  5,  9,  5],
              [3,  96, 3,  25, 12, 3],
              [45, 82, 6,  63, 9,  26],
              [38, 6,  5,  62, 84, 9],
              [57, 3,  80, 2,  4,  2],
              [90, 6,  3,  1,  68, 60],
              [5,  3,  9,  29, 67, 77],
              [80, 7,  4,  38, 9,  8]])
B = np.array([36, 3, 9, 8, 7, 1, 11, 5])
C = np.array([6, 8, 3, 1, 9, 5])

# создаём симплекс таблицу
# для прямой
F = np.vstack((
    np.hstack((-C, np.zeros(len(B)), 0)),
    np.hstack((A, np.eye(len(B)), np.expand_dims(B, axis=1)))
))

# для двойственной
f = np.vstack((
    np.hstack((np.zeros(len(B)), np.zeros(len(C)), np.ones(len(C)), 0)),
    np.hstack((A.T, -np.eye(len(C)), np.eye(len(C)), np.expand_dims(C, axis=1)))
))


# симплекс метод
def simplex_method(T, m, n):
    j_0 = list(T[0][:-1]).index(min(T[0][:-1]))  # разрешающий столбец

    bn = []
    for i in range(m):
        if T[i][j_0] == 0:
            bn.append(0.)
        else:
            bn.append(T[i][-1]/T[i][j_0])
    i_0 = bn.index(min([x for x in bn[1:] if x > 0]))  # разрешающая строка

    s[i_0 - 1] = j_0  # замена базисного столбца
    el = T[i_0][j_0]  # разрешающий элемент

    print(T)
    print("разрешающая строка:", i_0, "\nразрешающий столбец:", j_0, "\nразрешающий элемент:", el)

    T[i_0] /= el  # делаем единицу из разрешающего элемента

    # шаг Жордана-Гаусса
    for i in range(m):
        if i != i_0:
            temp = T[i_0] * T[i][j_0]
            T[i] -= temp

    # сокращение слишком приближенных к нулю чисел
    # для исключения зацикливания
    for i in range(m):
        for j in range(n):
            if math.fabs(T[i][j]) < 10**(-10):
                T[i][j] = 0


# условие остановки
def solve(T, m, n):
    flag = True
    step = 0
    while flag:
        if np.min(T[0, :-1]) >= 0:  # Пока в нулевой строке не останется отрицательных элементов
            flag = False
        else:
            # отслеживание шагов, если нужно
            step += 1
            print("\nШаг:", step)
            simplex_method(T, m, n)


# печать решения
def print_solution(T, m, s):
    print("Решение алгоритма:")
    for i in range(m):
        if i in s:
            print("x"+str(i+1)+" = %.6f" % T[s.index(i)+1][-1])
        else:
            print("x"+str(i+1)+" = 0.000000")
    print("оптимальное значение функции = %.5f" % (T[0][-1]))


# решение
# для прямой
print("Решение прямой задачи симплекс методом")
print("Начальная таблица:")
print(F)
(m, n) = F.shape
s = list(range(n-m, n-1))  # Базовый список переменных
solve(F, m, n)
print('\nПосле последнего шага:\n', f)
print_solution(F, len(C), s)

# для двойственной
# сначала решаем дополнительную к двойственной
print("\nПоиск начальной угловой точки в двойственной задачи:")
print("Начальная, приведенная матрица:")
print(f)
f = np.vstack((
    np.hstack((np.zeros(len(B)), np.zeros(len(C)), np.ones(len(C)), 0)) - list(map(sum, zip(*f[1:]))),
    np.hstack((A.T, -np.eye(len(C)), np.eye(len(C)), np.expand_dims(C, axis=1)))
))
print("\nИзбавляемся от единиц для базисных векторов:")
print(f)
(m, n) = f.shape
s = list(range(n-m, n-1))  # Базовый список переменных
solve(f, m, n)
print('\nПосле последнего шага:\n', f)
print_solution(f, len(B), s)

print("\nРешение двойственной задачи симплекс методом")
print("Начальная, приведенная матрица:")
f = np.vstack((
    np.hstack((B, np.zeros(len(C)), 0)),
    np.hstack((f[1:m, :m + 7], np.expand_dims(f[1:m, -1], axis=1)))
))
print(f)
(m, n) = f.shape
for j in range(n):
    if sum(f[1:, j]) == 1:
        for i in range(1, m):
            if f[i, j] == 1:
                tmp = f[i] * f[0, j]
                f[0] -= tmp
print(f)

(m, n) = f.shape
s = list(range(n-m, n-1))  # Базовый список переменных
solve(f, m, n)
print('\nПосле последнего шага:\n', f)
print_solution(f, len(B), s)


# встроенная проверка
# к прямой
c = -C.copy()
A_ub = A.copy()
B_ub = B.copy()
res = optimize.linprog(c, A_ub, B_ub)
print("\nВстроенное решение прямой задачи:", "\nзначение функции =", res.fun, "\nзначение x =", res.x)

# к двойственной
c_ = B.copy()
A_ub_ = -A.T
B_ub_ = -C.copy()
res_ = optimize.linprog(c_, A_ub_, B_ub_)
print("\nВстроенное решение двойственной задачи:", "\nзначение функции =", res_.fun, "\nзначение y =", res_.x)
