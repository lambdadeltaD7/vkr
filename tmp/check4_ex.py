import numpy as np

def lagrange_derivatives(x, x2, x3):
    """
    L_k'(x) для узлов [-1, x2, x3, 1].
    """

    L1 = -(
        x2 * x3
        - 2 * x2 * x
        + x2
        - 2 * x3 * x
        + x3
        + 3 * x**2
        - 2 * x
    ) / (
        2 * (x2 + 1) * (x3 + 1)
    )

    # ИСПРАВЛЕНО
    L2 = (
        -2 * x3 * x
        + 3 * x**2
        - 1
    ) / (
        (x2 - 1)
        * (x2 + 1)
        * (x2 - x3)
    )

    # ИСПРАВЛЕНО
    L3 = (
        2 * x2 * x
        - 3 * x**2
        + 1
    ) / (
        (x2 - x3)
        * (x3 - 1)
        * (x3 + 1)
    )

    L4 = (
        x2 * x3
        - 2 * x2 * x
        - x2
        - 2 * x3 * x
        - x3
        + 3 * x**2
        + 2 * x
    ) / (
        2 * (x2 - 1) * (x3 - 1)
    )

    return np.array([L1, L2, L3, L4])

def solve_system(x, alpha, tol=1e-12):
    """
    Решение системы для заданных x и alpha.

    Возвращает список допустимых решений.
    """

    # ----------------------------------------
    # Квадратное уравнение:
    #
    # A*u^2 + B*u + C = 0
    # ----------------------------------------

    A = 2 * x

    B = (
        2 * alpha * x
        - 6 * x**2
        + 2
    )

    C = (
        -3 * alpha * x**2
        + alpha
        + 4 * x**3
        - 2 * x
    )

    # ----------------------------------------
    # Находим корни
    # ----------------------------------------

    if abs(A) < tol:

        # Тогда уравнение становится
        #
        # B*u + C = 0

        if abs(B) < tol:
            raise ValueError(
                "Вырожденный случай: A = B = 0."
            )

        roots = [-C / B]

    else:

        discriminant = B**2 - 4 * A * C

        if discriminant < -tol:
            return []

        discriminant = max(discriminant, 0.0)

        sqrt_D = np.sqrt(discriminant)

        roots = [
            (-B - sqrt_D) / (2 * A),
            (-B + sqrt_D) / (2 * A),
        ]

    # ----------------------------------------
    # Формируем решения
    # ----------------------------------------

    solutions = []

    for u in roots:

        v = u + alpha

        # Допустимость узлов
        if not (-1 < u < v < 1):
            continue

        # Производные Лагранжа
        L1, L2, L3, L4 = lagrange_derivatives(
            x, u, v
        )

        # D
        D = L1 - L2 + L3 - L4

        if abs(D) < tol:
            continue

        # Веса
        omega1 = L1 / D
        omega2 = -L2 / D
        omega3 = L3 / D
        omega4 = -L4 / D

        omega = np.array([
            omega1,
            omega2,
            omega3,
            omega4
        ])

        # lambda*
        lambda_star = (
            -omega1
            -omega2 * u
            +omega3 * v
            -omega4
        )

        solutions.append({
            "x_nodes": np.array([
                -1.0,
                u,
                v,
                1.0
            ]),

            "weights": omega,

            "lambda": lambda_star,
        })

    return solutions

def residual(x, alpha, solution):
    """
    Невязка исходной системы.

    Возвращает:
        components -- невязки отдельных уравнений
        max_abs    -- максимальная абсолютная невязка
        norm_2     -- евклидова норма невязки
    """

    nodes = solution["x_nodes"]
    omega = solution["weights"]
    lambda_star = solution["lambda"]

    x1, x2, x3, x4 = nodes
    omega1, omega2, omega3, omega4 = omega

    # Исходная система

    r0 = (
        omega1
        - omega2
        + omega3
        - omega4
    )

    r1 = (
        -omega1
        -omega2 * x2
        +omega3 * x3
        -omega4
        -lambda_star
    )

    r2 = (
        omega1
        -omega2 * x2**2
        +omega3 * x3**2
        -omega4
        -2 * lambda_star * x
    )

    r3 = (
        -omega1
        -omega2 * x2**3
        +omega3 * x3**3
        -omega4
        -3 * lambda_star * x**2
    )

    r4 = (
        omega1
        -omega2 * x2**4
        +omega3 * x3**4
        -omega4
        -4 * lambda_star * x**3
    )

    # Нормировка
    r_norm = (
        omega1
        +omega2
        +omega3
        +omega4
        -1
    )

    # x3 - x2 = alpha
    r_alpha = x3 - x2 - alpha

    components = np.array([
        r0,
        r1,
        r2,
        r3,
        r4,
        r_norm,
        r_alpha,
    ])

    return {
        "components": components,
        "max_abs": np.max(np.abs(components)),
        "norm_2": np.linalg.norm(components),
    }

x=float(input("x="))
alpha=float(input("alpha="))

solutions = solve_system(x, alpha)

for i, sol in enumerate(solutions, 1):

    print(f"\n=== Solution {i} ===")

    print("x1, x2, x3, x4:")
    print(sol["x_nodes"])

    print("\nomega:")
    print(sol["weights"])
    print(sum(sol["weights"]))

    print("\nlambda*:")
    print(sol["lambda"])

    r = residual(x, alpha, sol)

    print("\nResidual components:")
    print(r["components"])

    print("\nmax |r|:")
    print(r["max_abs"])

    print("\nL2 residual:")
    print(r["norm_2"])
