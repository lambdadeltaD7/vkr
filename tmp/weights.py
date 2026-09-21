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
        (-1) ** (k ) * derivatives[k]
        for k in range(m)
    )

    # omega_k
    omegas = [
        (-1) ** (k ) * derivatives[k] / denominator
        for k in range(m)
    ]

    return omegas



x = -10.3
m = 5
xs = [-1, -1/2**0.5, 0, 1/2**0.5, 1]


omegas = compute_omegas(x, m, *xs)
s2 = 2**0.5
print(omegas)
print([
    (4*x**3 - 3*x**2 - x + 1/2) / (16*x*(2*x**2-1)),
    -(-8*x**3 + 3*s2*x**2 + 4*x - s2) / (16*x*(2*x**2-1)),
    (8*x**3 - 6*x) / (16*x*(2*x**2-1)),
    -(-8*x**3 - 3*s2*x**2 + 4*x + s2) / (16*x*(2*x**2-1)),
    (4*x**3 + 3*x**2 - x - 1/2) / (16*x*(2*x**2-1))
])


