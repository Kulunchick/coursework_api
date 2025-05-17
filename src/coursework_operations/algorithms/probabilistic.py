import numpy as np
from src.coursework_operations.utils.strategy import SolverStrategy
from src.coursework_operations.models.task import Task


class ProbabilisticAssignmentSolver(SolverStrategy):
    def __init__(self, Kmax=1000):
        super().__init__()
        self.Kmax = Kmax

    async def solve(self, task: Task):
        task.validate()

        v = np.where(task.B_ij != 0, task.c * (1 - task.omega) / task.B_ij, 0)

        Fbest = 0
        xbest = np.zeros((task.m, task.n), dtype=int)

        for iteration in range(self.Kmax):
            x = np.zeros((task.m, task.n), dtype=int)
            Tused = 0

            while True:
                allowed = (x == 0) & (Tused + task.B_ij <= task.B_total)
                values = v * allowed

                if not np.any(allowed):
                    break

                probs = values / values.sum()
                idx = np.unravel_index(np.random.choice(task.m * task.n, p=probs.flatten()), (task.m, task.n))

                x[idx] = 1
                Tused += task.B_ij[idx]

            F = np.sum((1 - task.omega) * task.c * x)

            if F > Fbest:
                Fbest = F
                xbest = x.copy()

            if self.iteration_callback:
                await self.iteration_callback({
                    'iteration': iteration + 1,
                    'current_best_value': Fbest
                })

        return xbest, Fbest
