import numpy as np

from src.coursework_operations.models.task import Task


class TaskGenerator:
    def __init__(self, c_min, c_max, b_min, b_max, omega_min, omega_max):
        self.c_min = c_min
        self.c_max = c_max
        self.b_min = b_min
        self.b_max = b_max
        self.omega_min = omega_min
        self.omega_max = omega_max

    def generate_task(self, m, n):
        c = np.random.uniform(self.c_min, self.c_max, (m, n)).tolist()

        B_ij = np.random.uniform(self.b_min, self.b_max, (m, n)).tolist()

        omega = np.random.uniform(self.omega_min, self.omega_max, (m, n)).tolist()

        B = np.random.randint(np.min(B_ij), np.sum(B_ij))
        # B = np.sum(B_ij) // 5

        task = Task(m=m, n=n, c=c, B_ij=B_ij, B_total=B, omega=omega)
        return task

    def generate_multiple_tasks(self, m, n, num_tasks):
        tasks = []
        for i in range(num_tasks):
            task = self.generate_task(m, n)
            tasks.append(task)
        return tasks

