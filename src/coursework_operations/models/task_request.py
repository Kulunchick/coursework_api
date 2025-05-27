from typing import List

from pydantic import BaseModel

from src.coursework_operations.models.algorithm_parametrs import AlgorithmParameters


class TaskRequest(BaseModel):
    m: int
    n: int
    c: List[List[int]]
    B_ij: List[List[int]]
    B_total: int
    omega: List[List[float]]
    algorithm_parameters: AlgorithmParameters = AlgorithmParameters()