from abc import ABC, abstractmethod

from src.coursework_operations.models.task import Task


class SolverStrategy(ABC):
    def __init__(self):
        self.iteration_callback = None

    def set_iteration_callback(self, callback):
        self.iteration_callback = callback

    @abstractmethod
    def solve(self, task: Task):
        pass
