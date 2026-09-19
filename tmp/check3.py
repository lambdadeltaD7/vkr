import numpy as np
import math

def check_solution(omega1, omega2, t1, t2,  beta, x):
    """
    Проверяет решение системы уравнений.
    
    Параметры:
    omega1, omega2, t1, t2, t3, beta : float
        Найденные значения переменных.
    x : float
        Значение x, для которого проверяется векторное уравнение.
        
    Возвращает:
    tuple
        (res_vec, res_eq1, res_eq2) - вектор невязок и скалярные невязки.
    """
    
    # 1. Векторное уравнение
    # Левая часть: beta * (0, 1, 2x, 3x^2)^T
    lhs_vec = beta * np.array([0, 1, 2*x, 3*x**2])
    
    # Векторы в правой части (с учетом ваших исправлений)
    v1 = np.array([1, -1, 1, -1])
    v2 = np.array([1, t1, t1**2, t1**3])
    v3 = np.array([1, t2, t2**2, t2**3])
    
    # Правая часть: -omega1 * v1 + omega2 * v2 - (1 - omega1 - omega2) * v3
    rhs_vec = -omega1 * v1 + omega2 * v2 - (1 - omega1 - omega2) * v3
    
    # Невязка векторного уравнения (поэлементно)
    res_vec = lhs_vec - rhs_vec
    
    # 2. Дополнительные уравнения
    res_eq1 = (t2 - t1) / (t1 + 1) - 2
    
    return res_vec, res_eq1



def calculate_variables(s, x):
    """
    Вычисляет переменные по формулам из изображения.
    
    Параметры:
    s : float
        Параметр s (задается пользователем).
    x : float
        Значение переменной x.
        
    Возвращает:
    tuple
        (t1, t2, omega1, omega2, beta)
    """
    sqrt7 = math.sqrt(7)
    
    # Вычисляем t1, t2, t3
    t1 = (1 + 4*x + s * sqrt7 * (x + 1)) / 3
    t2 = 3 + 4*x + s * sqrt7 * (x + 1)
    
    # Вычисляем omega1, omega2
    omega1 = 2 * (4 + s * sqrt7) / 27
    omega2 = 0.5  # 1/2
    
    # Вычисляем beta
    beta =  (10 + 7 * s * sqrt7) * (x + 1) / 27
    
    return t1, t2, omega1, omega2, beta

s_val = float(input("s="))   # <-- Впишите сюда ваше значение s
x_val = float(input("x="))   # <-- Впишите сюда значение x (если оно неизвестно, можно подобрать позже)

t1_val, t2_val, omega1_val, omega2_val, beta_val = calculate_variables(s_val, x_val)

# omega1_val = 0.2   
# omega2_val = 0.5   
# t1_val = 1.0       
# t2_val = 3.0       
# t3_val = 5.0       
# beta_val = 1.5     
# x_val = 2.0        # Значение x, для которого проверяем (если x неизвестен, попробуйте разные значения)

res_vec, res_eq1 = check_solution(omega1_val, omega2_val, t1_val, t2_val, beta_val, x_val)

print("--- Результаты проверки ---")
print("Невязка векторного уравнения (LHS - RHS):")
print(f"  Компонента 1: {res_vec[0]:.6f}")
print(f"  Компонента 2: {res_vec[1]:.6f}")
print(f"  Компонента 3: {res_vec[2]:.6f}")
print(f"  Компонента 4: {res_vec[3]:.6f}")

print(f"\nНевязка уравнения (t3-t2)/(t1+1) = 1: {res_eq1:.6f}")

# Суммарная квадратичная невязка (норма вектора невязок + квадраты скалярных невязок)
total_residual = np.sum(res_vec**2) + res_eq1**2 

