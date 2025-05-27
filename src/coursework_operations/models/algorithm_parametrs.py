from pydantic import BaseModel


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