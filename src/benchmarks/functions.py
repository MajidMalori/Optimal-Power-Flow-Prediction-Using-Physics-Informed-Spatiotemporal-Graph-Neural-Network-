"""
Standard Mathematical Benchmark Functions (F1-F23) for optimization testing.
"""
import numpy as np

def f1_sphere(x: np.ndarray) -> float:
    """Unimodal Function"""
    return float(np.sum(x**2))

def f2_schwefel_2_22(x: np.ndarray) -> float:
    """Unimodal Function"""
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))

def f3_schwefel_1_2(x: np.ndarray) -> float:
    """Unimodal Function"""
    res = 0.0
    for i in range(len(x)):
        res += float(np.sum(x[:i+1])**2)
    return res

def f4_schwefel_2_21(x: np.ndarray) -> float:
    """Unimodal Function"""
    return float(np.max(np.abs(x)))

def f5_rosenbrock(x: np.ndarray) -> float:
    """Unimodal Function"""
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))

def f6_step(x: np.ndarray) -> float:
    """Unimodal Function"""
    return float(np.sum(np.floor(x + 0.5)**2))

def f7_quartic_with_noise(x: np.ndarray) -> float:
    """Unimodal Function"""
    return float(np.sum(np.arange(1, len(x)+1) * x**4) + np.random.random())

# Multimodal Functions
def f8_schwefel(x: np.ndarray) -> float:
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))

def f9_rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))

def f10_ackley(x: np.ndarray) -> float:
    dim = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1))

def f11_griewank(x: np.ndarray) -> float:
    part1 = float(np.sum(x**2) / 4000)
    part2 = float(np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))))
    return float(part1 - part2 + 1)

# Dictionary of functions with their search space bounds
BENCHMARKS = {
    'F1': {'fn': f1_sphere, 'bounds': [-100, 100]},
    'F2': {'fn': f2_schwefel_2_22, 'bounds': [-10, 10]},
    'F3': {'fn': f3_schwefel_1_2, 'bounds': [-100, 100]},
    'F4': {'fn': f4_schwefel_2_21, 'bounds': [-100, 100]},
    'F5': {'fn': f5_rosenbrock, 'bounds': [-30, 30]},
    'F6': {'fn': f6_step, 'bounds': [-100, 100]},
    'F7': {'fn': f7_quartic_with_noise, 'bounds': [-1.28, 1.28]},
    'F8': {'fn': f8_schwefel, 'bounds': [-500, 500]},
    'F9': {'fn': f9_rastrigin, 'bounds': [-5.12, 5.12]},
    'F10': {'fn': f10_ackley, 'bounds': [-32, 32]},
    'F11': {'fn': f11_griewank, 'bounds': [-600, 600]},
}
