import numpy as np
import math

def check_solution(omega,  x,  beta, x_point):
    """
    Проверяет решение системы уравнений.
    
    Параметры:
    omega  t3, beta : float
        Найденные значения переменных.
    x : float
        Значение x, для которого проверяется векторное уравнение.
        
    Возвращает:
    tuple
        (res_vec, res_eq1, res_eq2) - вектор невязок и скалярные невязки.
    """
    
    # 1. Векторное уравнение
    lhs_vec = beta * np.array([0, 1, 2*x_point, 3*x_point**2, 4*x_point**3])
    
    # Векторы в правой части (с учетом ваших исправлений)
    v = [ np.array([1, x[i], x[i]**2, x[i]**3, x[i]**4]) for i in range(4)]
    
    # Правая часть: -omega1 * v1 + omega2 * v2 - (1 - omega1 - omega2) * v3
    rhs_vec = 0
    for i in range(4):
        rhs_vec += (-1)**(i) * omega[i] * v[i]
    
    # Невязка векторного уравнения (поэлементно)
    res_vec = lhs_vec - rhs_vec
    
    return res_vec


def compute_omegas(x, m, *xs):
    if len(xs) != m:
        raise ValueError(f"Expected {m} nodes, got {len(xs)}")

    # L_k(x)
    def L(k):
        result = 1.0
        for j in range(m):
            if j != k:
                result *= (x - xs[j]) / (xs[k] - xs[j])
        return result

    # L'_k(x)
    def L_derivative(k):
        # Если x совпадает с одним из узлов, формула
        # L_k(x) * sum(...) может дать 0 * inf,
        # поэтому считаем производную напрямую.
        result = 0.0

        for r in range(m):
            if r == k:
                continue

            term = 1.0 / (xs[k] - xs[r])

            for j in range(m):
                if j != k and j != r:
                    term *= (x - xs[j]) / (xs[k] - xs[j])

            result += term

        return result

    # Считаем все L'_k(x)
    derivatives = [
        L_derivative(k)
        for k in range(m)
    ]

    # Знаменатель
    denominator = sum(
        (-1) ** (k) * derivatives[k]
        for k in range(m)
    )

    # omega_k
    omegas = [
        (-1) ** (k) * derivatives[k] / denominator
        for k in range(m)
    ]

    return omegas,denominator





def s(k):
    return (
        -1 / 4
        + math.sqrt(7 / 12)
        * math.cos(
            math.acos(3 * math.sqrt(21) / 49) / 3
            + 2 * math.pi * k / 3
        )
    )

def calculate_variables(s, x_point):
    b = (x_point - s) / (s + 1) 
    a = b + 1
    print(f"a={a}")

    x = [ 
        a * math.cos( ((4-i) * math.pi) / 4 ) + b for i in range(4)
    ]
    

    omega,denom = compute_omegas(x_point, 4, *x)
    return x, omega, 1/denom

x_point = float(input("x_point="))   # <-- Впишите сюда значение x (если оно неизвестно, можно подобрать позже)

for k in range(3):
    x_val, omega_val, beta_val = calculate_variables(s(k), x_point)

    res_vec = check_solution(omega_val, x_val, beta_val, x_point)
    
    print(f"s[k]={s(k)}")
    print(f"x = {x_val}")
    print(f"omega = {omega_val}")
    print(f"sum(omega)={sum(omega_val)}")
    print(f"lambda = {beta_val}")
    print("--- Результаты проверки ---")
    print("Невязка векторного уравнения (LHS - RHS):")
    print(f"  Компонента 1: {res_vec[0]:.6f}")
    print(f"  Компонента 2: {res_vec[1]:.6f}")
    print(f"  Компонента 3: {res_vec[2]:.6f}")
    print(f"  Компонента 4: {res_vec[3]:.6f}")
    print(f"  Компонента 5: {res_vec[4]:.6f}")
    print()
    print()


