"""
WaferGuard: Production-grade wafer anomaly detection pipeline.
Fixes critical bugs in the original bnypi.ipynb notebook.
"""

__version__ = "1.0.0"

from . import data
from . import preprocessing
from . import model
from . import training
from . import evaluation

__all__ = ["data", "preprocessing", "model", "training", "evaluation"]
