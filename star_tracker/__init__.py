"""
star_tracker – automated Clash of Clans war-star parser
"""
# ---------------------------------------------------------------------------
# Imports
from importlib.metadata import version, PackageNotFoundError

# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

from .data_structures import (
    dataColumn,
    attackData,
    playerData,
)
from .main import main

__all__ = [
    "dataColumn",
    "attackData",
    "playerData",
    "main",
    "__version__",
]

try:
    __version__ = version("star_tracker")
except PackageNotFoundError:        # running from a checkout / no wheel yet
    __version__ = "0.0.0.dev0"