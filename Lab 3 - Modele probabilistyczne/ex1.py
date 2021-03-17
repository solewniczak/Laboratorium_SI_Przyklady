from itertools import product
import numpy as np
import matplotlib.pyplot as plt

F = 0
P = 1

p_k = np.zeros(2)
p_k[P] = 0.8
p_k[F] = 0.2
p_s = np.zeros(2)
p_s[P] = 0.02
p_s[F] = 0.98

pw_w_k = np.zeros((2, 2))
pw_w_k[P, P] = 0.9
pw_w_k[P, F] = 0.1
pw_w_k[F, P] = 0.01
pw_w_k[F, F] = 0.99

pw_u_ks = np.zeros((2, 2, 2))
pw_u_ks[P, P, P] = 0.9
pw_u_ks[P, P, F] = 0.1
pw_u_ks[P, F, P] = 0.2
pw_u_ks[P, F, F] = 0.8
pw_u_ks[F, P, P] = 0.9
pw_u_ks[F, P, F] = 0.1
pw_u_ks[F, F, P] = 0.01
pw_u_ks[F, F, F] = 0.99

pw_b_u = np.zeros((2, 2))
pw_b_u[P, P] = 0.7
pw_b_u[P, F] = 0.3
pw_b_u[F, P] = 0.1
pw_b_u[F, F] = 0.9

# Obliczanie wybranych prawdopodobieństw warunkowch metodą dokładną

# Generowanie tablicy łącznego rozkładu pradopodobieństwa
p = np.zeros((2, 2, 2, 2, 2))
for K in [P, F]:
    for S in [P, F]:
        for W in [P, F]:
            for U in [P, F]:
                for B in [P, F]:
                    p[K, S, W, U, B] = p_k[K]*p_s[S]*pw_w_k[K, W]*pw_u_ks[K, S, U]*pw_b_u[U, B]

# P(B = P|K = P) = P(B = P, K = P)/P(K = P)
# P(B = P, K = P)
pw_b_and_k = sum([p[P, s, w, u, P] for s, w, u in product([P, F], [P, F], [P, F])])
# P(K = P)
pw_k = sum([p[P, s, w, u, b] for s, w, u, b in product([P, F], [P, F], [P, F], [P, F])])
pw_b_k_exact = pw_b_and_k/pw_k
print('P(B = P|K = P) exact: ', pw_b_k_exact)

# P(K = P|B = P) = P(K = P, B = P)/P(B = P)
# P(K = P, B = P)
pw_k_and_b = sum([p[P, s, w, b, P] for s, w, b in product([P, F], [P, F], [P, F])])
# P(B = P)
pw_b = sum([p[k, s, w, u, P] for k, s, w, u in product([P, F], [P, F], [P, F], [P, F])])
pw_k_b_exact = pw_k_and_b/pw_b
print('P(K = P|B = P) exact: ', pw_k_b_exact)


# Obliczanie wybranych prawdopodobieństw warunkowch metodą Monte Carlo.

I = 100 # liczba iteracji
K = 300 # liczba próbek na iterację

avg_pw_b_k = 0
avg_pw_k_b = 0

plt.axis([0, I, 0, 1])
plt.xlabel('Iteracja')
plt.ylabel('Prawdopodobieństwo')
plt.plot([0, I], [pw_b_k_exact, pw_b_k_exact], color='b')
plt.plot([0, I], [pw_k_b_exact, pw_k_b_exact], color='b')

np.random.seed(1)
for i in range(1, I):
    # Wygenerowanie wartości poszczególnych zmiennych losowych dla losowego przebiegu sieci
    k = np.random.random(K) < p_k[P]
    s = np.random.random(K) < p_s[P]
    uPP = np.random.random(K) < pw_u_ks[P, P, P]
    uPF = np.random.random(K) < pw_u_ks[P, F, P]
    uFP = np.random.random(K) < pw_u_ks[F, P, P]
    uFF = np.random.random(K) < pw_u_ks[F, F, P]
    u = np.logical_or.reduce((
        np.logical_and.reduce((k, s, uPP)),
        np.logical_and.reduce((k, np.logical_not(s), uPF)),
        np.logical_and.reduce((np.logical_not(k), s, uFP)),
        np.logical_and.reduce((np.logical_not(k), np.logical_not(s), uFF)),
    ))

    bP = np.random.random(K) < pw_b_u[P, P]
    bF = np.random.random(K) < pw_b_u[F, P]
    b = np.logical_or(
        np.logical_and(u, bP),
        np.logical_and(np.logical_not(u), bF)
    )

    # Oszacowanie pradopodobieństw warunkowych na podstawie tablic:  k, s, u, b

    # P(B = P|K = P) = P(B = P, K = P)/P(K = P)
    pw_b_k__mc = np.sum(np.logical_and(b, k)) / np.sum(k)

    # P(K = P|B = P) = P(K = P, B = P)/P(B = P)
    pw_k_b__mc = np.sum(np.logical_and(k, b))/np.sum(b)

    # Ze wzoru na iteracyjne liczenie średniej
    avg_pw_b_k = avg_pw_b_k + (pw_b_k__mc - avg_pw_b_k) / i
    avg_pw_k_b = avg_pw_k_b + (pw_k_b__mc - avg_pw_k_b) / i

    plt.scatter(i, avg_pw_b_k, marker='.', s=1, color='r')
    plt.scatter(i, avg_pw_k_b, marker='.', s=1, color='r')

    # Poniższą linię można odkomentować w celu śledzenia działania metody krok po kroku
    plt.pause(0.001)

print('P(B = P|K = P) Monte Carlo: ', avg_pw_b_k)
print('P(K = P|B = P) Monte Carlo: ', avg_pw_k_b)