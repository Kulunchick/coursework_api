import asyncio
import time
from typing import List

import numpy as np
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from src.coursework_operations.algorithms.ant_colony import AntColonyAssignmentSolver
from src.coursework_operations.algorithms.probabilistic import ProbabilisticAssignmentSolver
from src.coursework_operations.models.task import Task
from src.coursework_operations.utils.context import Context
from src.coursework_operations.utils.generator import TaskGenerator
from assignment_solver import AntColonyAssignmentSolver as RustAntColony
from assignment_solver import ProbabilisticAssignmentSolver as RustProbabilistic
from assignment_solver import Task as RustTask


app = FastAPI()


class AntColonyParameters(BaseModel):
    Kmax: int = 500
    num_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    p: float = 0.1
    tau: float = 1.0


class ProbabilisticParameters(BaseModel):
    Kmax: int = 1000


class AlgorithmParameters(BaseModel):
    ant_colony: AntColonyParameters = AntColonyParameters()
    probabilistic: ProbabilisticParameters = ProbabilisticParameters()


class TaskRequest(BaseModel):
    m: int
    n: int
    c: List[List[int]]
    B_ij: List[List[int]]
    B_total: int
    omega: List[List[float]]
    algorithm_parameters: AlgorithmParameters = AlgorithmParameters()


async def send_iteration_data(websocket: WebSocket, data: dict):
    await websocket.send_json({
        "type": "iteration",
        **data
    })


class AlphaVariant(BaseModel):
    alpha: float
    beta: float


class MNVariant(BaseModel):
    m: int
    n: int


class KmaxVariant(BaseModel):
    kmax: int


class LVariant(BaseModel):
    l: int


class Range(BaseModel):
    min: float
    max: float


class ExperimentRequest(BaseModel):
    count: int
    variants: List[AlphaVariant]
    mnVariants: List[MNVariant]
    kmaxVariants: List[KmaxVariant]
    lVariants: List[LVariant]
    p: float
    tau: float
    cRange: Range
    bRange: Range
    omegaRange: Range


@app.websocket("/ws/solve")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        task_data = TaskRequest(**data)

        task = Task(
            m=task_data.m,
            n=task_data.n,
            c=task_data.c,
            B_ij=task_data.B_ij,
            B_total=task_data.B_total,
            omega=task_data.omega
        )

        context = Context()

        # Створюємо завдання для паралельного виконання алгоритмів
        async def run_algorithm(algorithm_name: str, solver):
            solver.set_iteration_callback(
                lambda data: asyncio.create_task(websocket.send_json({
                    "type": "iteration",
                    "algorithm": algorithm_name,
                    **data
                }))
            )

            context.strategy = solver

            # Повідомляємо про початок виконання алгоритму
            await websocket.send_json({"type": "start", "algorithm": algorithm_name})

            # Виконуємо алгоритм
            solution, value = await context.execute_strategy(task)

            # Відправляємо фінальний результат
            await websocket.send_json({
                "type": "result",
                "algorithm": algorithm_name,
                "solution": solution.tolist(),
                "value": float(value)
            })

        # Додати мурашиний алгоритм
        ant_colony = AntColonyAssignmentSolver(
            Kmax=task_data.algorithm_parameters.ant_colony.Kmax,
            num_ants=task_data.algorithm_parameters.ant_colony.num_ants,
            alpha=task_data.algorithm_parameters.ant_colony.alpha,
            beta=task_data.algorithm_parameters.ant_colony.beta,
            evaporation_rate=task_data.algorithm_parameters.ant_colony.p,
            initial_pheromone=task_data.algorithm_parameters.ant_colony.tau,
        )

        # Додати ймовірнісний алгоритм
        prob = ProbabilisticAssignmentSolver(Kmax=task_data.algorithm_parameters.probabilistic.Kmax)

        # Запуск паралельного виконання
        await asyncio.gather(
            run_algorithm("ant_colony", ant_colony),
            run_algorithm("probabilistic", prob)
        )

        # Повідомляємо про завершення
        await websocket.send_json({"type": "complete"})

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


@app.websocket("/ws/experiment")
async def experiment_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        experiment_data = ExperimentRequest(**data)

        task_generator = TaskGenerator(
            c_min=experiment_data.cRange.min,
            c_max=experiment_data.cRange.max,
            b_min=experiment_data.bRange.min,
            b_max=experiment_data.bRange.max,
            omega_min=experiment_data.omegaRange.min,
            omega_max=experiment_data.omegaRange.max
        )

        # Перебираємо всі комбінації параметрів
        for alpha_variant in experiment_data.variants:
            for mn_variant in experiment_data.mnVariants:
                for kmax_variant in experiment_data.kmaxVariants:
                    for l_variant in experiment_data.lVariants:
                        tasks = task_generator.generate_multiple_tasks(
                            m=mn_variant.m,
                            n=mn_variant.n,
                            num_tasks=experiment_data.count
                        )

                        # Створюємо Rust-солвери з поточними параметрами
                        rust_ant_colony = RustAntColony(
                            num_ants=l_variant.l,
                            kmax=kmax_variant.kmax,
                            alpha=alpha_variant.alpha,
                            beta=alpha_variant.beta,
                            rho=experiment_data.p,
                            initial_pheromone=experiment_data.tau
                        )

                        rust_probabilistic = RustProbabilistic(kmax=kmax_variant.kmax)

                        results = []
                        # Запускаємо експерименти для кожної задачі
                        for task_index, task in enumerate(tasks):
                            rust_task = RustTask(
                                task.m,
                                task.n,
                                task.c,
                                task.B_ij,
                                int(task.B_total),
                                task.omega
                            )

                            ant_start_time = time.perf_counter()
                            ant_solution, ant_value = rust_ant_colony.solve(rust_task)
                            ant_time = time.perf_counter() - ant_start_time

                            # Заміри часу для ймовірнісного алгоритму
                            prob_start_time = time.perf_counter()
                            prob_solution, prob_value = rust_probabilistic.solve(rust_task)
                            prob_time = time.perf_counter() - prob_start_time

                            results.append({
                                "ant_colony": {
                                    "value": float(ant_value),
                                    "time": ant_time
                                },
                                "probabilistic": {
                                    "value": float(prob_value),
                                    "time": prob_time
                                }
                            })

                        f_prob = np.mean([r["probabilistic"]["value"] for r in results])
                        f_ant = np.mean([r["ant_colony"]["value"] for r in results])

                        await websocket.send_json({
                            "type": "complete",
                            "results": [
                                mn_variant.m,
                                mn_variant.n,
                                alpha_variant.alpha,
                                alpha_variant.beta,
                                kmax_variant.kmax,
                                l_variant.l,
                                np.mean([r["probabilistic"]["time"] for r in results]),
                                f_prob,
                                np.mean([r["ant_colony"]["time"] for r in results]),
                                f_ant,
                                (f_ant - f_prob) / f_prob * 100,
                            ]
                        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
