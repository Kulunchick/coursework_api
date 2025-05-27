import asyncio
from fastapi import WebSocket
from src.coursework_operations.algorithms.ant_colony import AntColonyAssignmentSolver
from src.coursework_operations.algorithms.probabilistic import ProbabilisticAssignmentSolver
from src.coursework_operations.models.task import Task
from src.coursework_operations.utils.context import Context
from src.coursework_operations.models.task_request import TaskRequest

async def solve_handler(websocket: WebSocket):
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

        async def run_algorithm(algorithm_name: str, solver):
            solver.set_iteration_callback(
                lambda data: asyncio.create_task(websocket.send_json({
                    "type": "iteration",
                    "algorithm": algorithm_name,
                    **data
                }))
            )

            context.strategy = solver
            await websocket.send_json({"type": "start", "algorithm": algorithm_name})
            solution, value = await context.execute_strategy(task)
            await websocket.send_json({
                "type": "result",
                "algorithm": algorithm_name,
                "solution": solution.tolist(),
                "value": float(value)
            })

        ant_colony = AntColonyAssignmentSolver(
            Kmax=task_data.algorithm_parameters.ant_colony.Kmax,
            num_ants=task_data.algorithm_parameters.ant_colony.num_ants,
            alpha=task_data.algorithm_parameters.ant_colony.alpha,
            beta=task_data.algorithm_parameters.ant_colony.beta,
            evaporation_rate=task_data.algorithm_parameters.ant_colony.p,
            initial_pheromone=task_data.algorithm_parameters.ant_colony.tau,
        )

        prob = ProbabilisticAssignmentSolver(
            Kmax=task_data.algorithm_parameters.probabilistic.Kmax
        )

        await asyncio.gather(
            run_algorithm("ant_colony", ant_colony),
            run_algorithm("probabilistic", prob)
        )

        await websocket.send_json({"type": "complete"})

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()