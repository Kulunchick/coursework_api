import time
from typing import List
import numpy as np
from assignment_solver import (
    AntColonyAssignmentSolver as RustAntColony,
    ProbabilisticAssignmentSolver as RustProbabilistic,
    Task as RustTask
)
from pydantic import BaseModel
from fastapi import WebSocket
from src.coursework_operations.utils.generator import TaskGenerator


class Range(BaseModel):
    min: float
    max: float

class ExperimentRequest(BaseModel):
    count: int
    omegaRangeVariants: List[Range]
    p: float
    tau: float
    alpha: float
    beta: float
    m: int
    n: int
    antKmax: int
    probKmax: int
    l: int
    cRange: Range
    bRange: Range


async def experiment4_handler(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        experiment_data = ExperimentRequest(**data)

        # Словники для зберігання результатів
        results = {
            f"{omega_range.min}-{omega_range.max}": {
                "ant": {
                    "values": [],
                    "times": []
                },
                "prob": {
                    "values": [],
                    "times": []
                }
            }
            for omega_range in experiment_data.omegaRangeVariants
        }

        for i in range(experiment_data.count):
            for omega_range in experiment_data.omegaRangeVariants:
                task_generator = TaskGenerator(
                    c_min=experiment_data.cRange.min,
                    c_max=experiment_data.cRange.max,
                    b_min=experiment_data.bRange.min,
                    b_max=experiment_data.bRange.max,
                    omega_min=omega_range.min,
                    omega_max=omega_range.max
                )

                task = task_generator.generate_task(
                    m=experiment_data.m,
                    n=experiment_data.n
                )

                rust_ant_colony = RustAntColony(
                    num_ants=experiment_data.l,
                    kmax=experiment_data.antKmax,
                    alpha=experiment_data.alpha,
                    beta=experiment_data.beta,
                    rho=experiment_data.p,
                    initial_pheromone=experiment_data.tau
                )

                rust_probabilistic = RustProbabilistic(kmax=experiment_data.probKmax)

                rust_task = RustTask(
                    task.m,
                    task.n,
                    task.c,
                    task.B_ij,
                    int(task.B_total),
                    task.omega
                )

                # Мурашиний алгоритм
                ant_start_time = time.perf_counter()
                ant_solution, ant_value = rust_ant_colony.solve(rust_task)
                ant_time = time.perf_counter() - ant_start_time

                # Ймовірнісний алгоритм
                prob_start_time = time.perf_counter()
                prob_solution, prob_value = rust_probabilistic.solve(rust_task)
                prob_time = time.perf_counter() - prob_start_time

                # Зберігаємо результати
                key = f"{omega_range.min}-{omega_range.max}"
                results[key]["ant"]["values"].append(ant_value)
                results[key]["ant"]["times"].append(ant_time)
                results[key]["prob"]["values"].append(prob_value)
                results[key]["prob"]["times"].append(prob_time)

        # Обчислюємо середні значення та відносну різницю
        final_results = {}
        for omega_range, data in results.items():
            ant_avg_value = float(np.mean(data["ant"]["values"]))
            ant_avg_time = float(np.mean(data["ant"]["times"]))
            prob_avg_value = float(np.mean(data["prob"]["values"]))
            prob_avg_time = float(np.mean(data["prob"]["times"]))

            # Обчислюємо відносну різницю у відсотках для значень цільової функції
            value_relative_diff = ((ant_avg_value - prob_avg_value) / ant_avg_value * 100
                                   if ant_avg_value != 0 else 0)

            final_results[omega_range] = {
                "ant": {
                    "avg_value": ant_avg_value,
                    "avg_time": ant_avg_time
                },
                "prob": {
                    "avg_value": prob_avg_value,
                    "avg_time": prob_avg_time
                },
                "relative_difference": value_relative_diff  # у відсотках
            }

        # Відправляємо результати
        await websocket.send_json({
            "type": "results",
            "data": final_results
        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()