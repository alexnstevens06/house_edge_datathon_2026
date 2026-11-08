"""Internal synthetic episode generators for research experiments.

This package is not an organizer schema and it does not read datathon files.
"""

from burst_refractory.generators import (
    N_MACHINES,
    PAYOUTS,
    ParameterError,
    generate_burst,
    generate_refractory,
)

__all__ = [
    "N_MACHINES",
    "PAYOUTS",
    "ParameterError",
    "generate_burst",
    "generate_refractory",
]
