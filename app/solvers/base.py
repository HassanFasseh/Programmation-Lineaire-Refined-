# solvers/base.py
from abc import ABC, abstractmethod
import numpy as np

class OptimizationSolver(ABC):
    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray, bounds, maximize: bool):
        self.c = c
        self.A = A
        self.b = b
        self.bounds = bounds
        self.maximize = maximize

    @abstractmethod
    def solve(self):
        pass
