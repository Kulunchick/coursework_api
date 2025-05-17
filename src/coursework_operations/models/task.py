from typing import List

import numpy as np


class Task:
    def __init__(self, m: int, n: int, c: List[List[int]], B_ij: List[List[int]], B_total: int, omega: List[List[float]]):
        self.m = m
        self.n = n
        self.c = np.array(c, dtype=np.int64)
        self.B_ij = np.array(B_ij, dtype=np.int64)
        self.B_total = B_total
        self.omega = np.array(omega, dtype=np.float64)


    def validate(self):
        if not all(len(row) == self.n for row in self.c):
            raise ValueError("Некоректні розміри матриці вартості")

        if not all(len(row) == self.n for row in self.B_ij):
            raise ValueError("Некоректні розміри матриці витрат ресурсу")

        if not all(len(row) == self.n for row in self.omega):
            raise ValueError("Некоректні розміри матриці знижок")

        if len(self.c) != self.m or len(self.B_ij) != self.m or len(self.omega) != self.m:
            raise ValueError("Кількість рядків у матрицях не відповідає кількості типів обладнання")

        if not all(0 <= x <= 1 for row in self.omega for x in row):
            raise ValueError("Знижки повинні бути в діапазоні [0, 1]")

        if self.B_total <= 0:
            raise ValueError("Загальний ресурс повинен бути додатнім")

        return True
