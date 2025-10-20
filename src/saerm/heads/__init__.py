from .base import HeadFactory, PredictionHead
from . import linear  # noqa: F401
from . import tree  # noqa: F401
from . import xgboost  # noqa: F401

try:  # noqa: F401
    from . import gam  # noqa: F401
except ImportError:  # pragma: no cover
    gam = None  # type: ignore

__all__ = ["HeadFactory", "PredictionHead"]
