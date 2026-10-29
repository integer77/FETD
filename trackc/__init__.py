"""Track C: GPU-ready mixed-precision frequency-domain Maxwell solver (design C1).

FP32 sparse factorization of the edge system, plus an FP32 nodal gradient
correction, wrapped in an FP64 residual refinement loop.
"""
from .factor import Factorization
from .solver import GradientCorrectedPrecond, SolveResult, refine, gmres_ir
from .sweep import FrequencySweep

__all__ = [
    "Factorization", "GradientCorrectedPrecond", "SolveResult",
    "refine", "gmres_ir", "FrequencySweep",
]
