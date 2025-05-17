from src.coursework_operations.utils.strategy import SolverStrategy
from src.coursework_operations.models.task import Task


class Context:
    def __init__(self, strategy: SolverStrategy = None):
        self._strategy = strategy

    @property
    def strategy(self) -> SolverStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: SolverStrategy):
        self._strategy = strategy

    def execute_strategy(self, task: Task):
        if self._strategy is None:
            raise ValueError("Стратегія не встановлена")
        if not task.validate():
            raise ValueError("Завдання не валідне")
        return self._strategy.solve(task)
