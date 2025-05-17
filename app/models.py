# models.py
from pydantic import BaseModel
from typing import List, Tuple, Optional

class OptimizationRequest(BaseModel):
    objective_coefficients: List[float]
    constraint_matrix: List[List[float]]
    rhs_values: List[float]
    variable_bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    method: str
    maximize: bool = True
