"""
Standard Mathematical Benchmark Functions (F1-F23) for optimization testing.
Reference: Yao, Liu & Lin (1999) - Evolutionary Programming Made Faster.
"""
import numpy as np


# ============================================================
# Unimodal Functions (F1-F7)
# ============================================================

def f1_sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def f2_schwefel_2_22(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))

def f3_schwefel_1_2(x: np.ndarray) -> float:
    res = 0.0
    for i in range(len(x)):
        res += float(np.sum(x[:i+1])**2)
    return res

def f4_schwefel_2_21(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

def f5_rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))

def f6_step(x: np.ndarray) -> float:
    return float(np.sum(np.floor(x + 0.5)**2))

def f7_quartic_with_noise(x: np.ndarray) -> float:
    return float(np.sum(np.arange(1, len(x)+1) * x**4) + np.random.random())


# ============================================================
# Multimodal Functions (F8-F13)
# ============================================================

def f8_schwefel(x: np.ndarray) -> float:
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))

def f9_rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))

def f10_ackley(x: np.ndarray) -> float:
    dim = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) -
           np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1))

def f11_griewank(x: np.ndarray) -> float:
    part1 = np.sum(x**2) / 4000
    part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(part1 - part2 + 1)

def _u_penalty(x, a, k, m):
    """Penalty helper for F12 and F13."""
    result = np.zeros_like(x)
    result[x > a] = k * (x[x > a] - a)**m
    result[x < -a] = k * (-x[x < -a] - a)**m
    return result

def f12_penalized_1(x: np.ndarray) -> float:
    dim = len(x)
    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2
    penalty = np.sum(_u_penalty(x, 10, 100, 4))
    return float((np.pi / dim) * (term1 + term2 + term3) + penalty)

def f13_penalized_2(x: np.ndarray) -> float:
    dim = len(x)
    term1 = np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    penalty = np.sum(_u_penalty(x, 5, 100, 4))
    return float(0.1 * (term1 + term2 + term3) + penalty)


# ============================================================
# Fixed-Dimension Multimodal Functions (F14-F23)
# ============================================================

def f14_shekel_foxholes(x: np.ndarray) -> float:
    a = np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],
                  [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
    total = 0.0
    for j in range(25):
        inner = (j + 1) + np.sum((x - a[:, j])**6)
        total += 1.0 / inner
    return float(1.0 / (0.002 + total))

def f15_kowalik(x: np.ndarray) -> float:
    a_vals = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b_vals = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    total = 0.0
    for i in range(11):
        num = x[0] * (b_vals[i]**2 + b_vals[i] * x[1])
        den = b_vals[i]**2 + b_vals[i] * x[2] + x[3]
        total += (a_vals[i] - num / den)**2
    return float(total)

def f16_six_hump_camel(x: np.ndarray) -> float:
    return float(4*x[0]**2 - 2.1*x[0]**4 + x[0]**6/3 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4)

def f17_branin(x: np.ndarray) -> float:
    a = 1; b = 5.1/(4*np.pi**2); c = 5/np.pi; r = 6; s = 10; t = 1/(8*np.pi)
    return float(a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s)

def f18_goldstein_price(x: np.ndarray) -> float:
    part1 = (1 + (x[0]+x[1]+1)**2 * (19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))
    part2 = (30 + (2*x[0]-3*x[1])**2 * (18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
    return float(part1 * part2)

def f19_hartmann_3(x: np.ndarray) -> float:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    P = 1e-4 * np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])
    total = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x[:3] - P[i])**2)
        total -= alpha[i] * np.exp(-inner)
    return float(total)

def f20_hartmann_6(x: np.ndarray) -> float:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],
                  [3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    P = 1e-4 * np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],
                          [2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
    total = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x[:6] - P[i])**2)
        total -= alpha[i] * np.exp(-inner)
    return float(total)

def _shekel(x: np.ndarray, m: int) -> float:
    a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    total = 0.0
    for i in range(m):
        inner = np.sum((x[:4] - a[i])**2) + c[i]
        total -= 1.0 / inner
    return float(total)

def f21_shekel_5(x: np.ndarray) -> float:
    return _shekel(x, 5)

def f22_shekel_7(x: np.ndarray) -> float:
    return _shekel(x, 7)

def f23_shekel_10(x: np.ndarray) -> float:
    return _shekel(x, 10)


# ============================================================
# Complete Benchmark Registry
# ============================================================
# 'dim' key: None = scalable (use runner's dim), int = fixed dimension

BENCHMARKS = {
    # Unimodal (F1-F7) - scalable dimension
    'F1':  {'fn': f1_sphere,             'bounds': [-100, 100],   'dim': None},
    'F2':  {'fn': f2_schwefel_2_22,      'bounds': [-10, 10],     'dim': None},
    'F3':  {'fn': f3_schwefel_1_2,       'bounds': [-100, 100],   'dim': None},
    'F4':  {'fn': f4_schwefel_2_21,      'bounds': [-100, 100],   'dim': None},
    'F5':  {'fn': f5_rosenbrock,         'bounds': [-30, 30],     'dim': None},
    'F6':  {'fn': f6_step,               'bounds': [-100, 100],   'dim': None},
    'F7':  {'fn': f7_quartic_with_noise, 'bounds': [-1.28, 1.28], 'dim': None},
    # Multimodal (F8-F13) - scalable dimension
    'F8':  {'fn': f8_schwefel,           'bounds': [-500, 500],   'dim': None},
    'F9':  {'fn': f9_rastrigin,          'bounds': [-5.12, 5.12], 'dim': None},
    'F10': {'fn': f10_ackley,            'bounds': [-32, 32],     'dim': None},
    'F11': {'fn': f11_griewank,          'bounds': [-600, 600],   'dim': None},
    'F12': {'fn': f12_penalized_1,       'bounds': [-50, 50],     'dim': None},
    'F13': {'fn': f13_penalized_2,       'bounds': [-50, 50],     'dim': None},
    # Fixed-dimension multimodal (F14-F23)
    'F14': {'fn': f14_shekel_foxholes,   'bounds': [-65.536, 65.536], 'dim': 2},
    'F15': {'fn': f15_kowalik,           'bounds': [-5, 5],           'dim': 4},
    'F16': {'fn': f16_six_hump_camel,    'bounds': [-5, 5],           'dim': 2},
    'F17': {'fn': f17_branin,            'bounds': [[-5, 10], [0, 15]], 'dim': 2},
    'F18': {'fn': f18_goldstein_price,   'bounds': [-2, 2],           'dim': 2},
    'F19': {'fn': f19_hartmann_3,        'bounds': [0, 1],            'dim': 3},
    'F20': {'fn': f20_hartmann_6,        'bounds': [0, 1],            'dim': 6},
    'F21': {'fn': f21_shekel_5,          'bounds': [0, 10],           'dim': 4},
    'F22': {'fn': f22_shekel_7,          'bounds': [0, 10],           'dim': 4},
    'F23': {'fn': f23_shekel_10,         'bounds': [0, 10],           'dim': 4},
}
