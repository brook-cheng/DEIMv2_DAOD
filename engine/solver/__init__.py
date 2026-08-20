"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver



from typing import Dict, Type

#: Solver factory registry: each value is the solver CLASS (a callable that
#: constructs a ``BaseSolver`` instance from a config), not a pre-built instance.
TASKS :Dict[str, Type[BaseSolver]] = {
    'classification': ClasSolver,
    'detection': DetSolver,
}
