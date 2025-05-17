import numpy as np
from src.coursework_operations.utils.strategy import SolverStrategy
from src.coursework_operations.models.task import Task


class AntColonyAssignmentSolver(SolverStrategy):
    def __init__(self, num_ants=20, Kmax=100, alpha=1.0, beta=2.0, evaporation_rate=0.1, initial_pheromone=1.0):
        super().__init__()
        self.num_ants = num_ants
        self.Kmax = Kmax
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.initial_pheromone = initial_pheromone

    async def solve(self, task: Task):
        pheromone = np.full((task.m, task.n), self.initial_pheromone)
        heuristic = np.where(task.B_ij != 0, (task.c * (1 - task.omega)) / task.B_ij, 0)

        Q = task.B_total / np.average(task.B_ij) * np.max(task.c)

        Fbest = 0
        xbest = np.zeros((task.m, task.n), dtype=int)

        for iteration in range(self.Kmax):
            all_ants_solutions = np.zeros((self.num_ants, task.m, task.n), dtype=int)
            all_ants_scores = np.zeros(self.num_ants)

            for ant in range(self.num_ants):
                x = np.zeros((task.m, task.n), dtype=int)
                Tused = 0

                while True:
                    allowed = (x == 0) & (Tused + task.B_ij <= task.B_total)
                    tau = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    tau *= allowed

                    if not np.any(allowed):
                        break

                    probs = tau / tau.sum()
                    idx = np.unravel_index(np.random.choice(task.m * task.n, p=probs.flatten()), (task.m, task.n))

                    x[idx] = 1
                    Tused += task.B_ij[idx]

                F = np.sum((1 - task.omega) * task.c * x)

                all_ants_solutions[ant] = x
                all_ants_scores[ant] = F

                if F > Fbest:
                    Fbest = F
                    xbest = x.copy()

            if self.iteration_callback:
                await self.iteration_callback({
                    'iteration': iteration + 1,
                    'current_best_value': Fbest
                })

            pheromone *= (1 - self.rho)
            delta_pheromone = np.zeros_like(pheromone)

            for ant in range(self.num_ants):
                delta_pheromone += all_ants_solutions[ant] * (all_ants_scores[ant] / Q)

            pheromone += delta_pheromone

        return xbest, Fbest
