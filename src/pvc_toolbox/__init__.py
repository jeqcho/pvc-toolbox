"""PVC Toolbox - Tools for computing the Proportional Veto Core.

This package provides efficient algorithms for computing the Proportional
Veto Core (PVC) and its epsilon relaxation (ε-PVC) for preference profiles.

Installation
------------
Basic installation:
    pip install pvc-toolbox

For faster computation with scipy:
    pip install pvc-toolbox[fast]

Usage
-----
>>> from pvc_toolbox import compute_pvc, is_in_pvc
>>> 
>>> # Define preferences as matrix: preferences[rank][voter]
>>> preferences = [
...     ["a", "b", "c"],  # rank 0 (most preferred)
...     ["b", "c", "a"],  # rank 1
...     ["c", "a", "b"],  # rank 2 (least preferred)
... ]
>>> alternatives = ["a", "b", "c"]
>>> 
>>> # Compute the full PVC
>>> pvc = compute_pvc(preferences, alternatives)
>>> print(pvc)
['a', 'b', 'c']
>>> 
>>> # Check if a specific alternative is in the PVC
>>> is_in_pvc(preferences, alternatives, "a")
True

References
----------
Egor Ianovski, Aleksei Y. Kondratev (2023). "Computing the proportional veto core".
"""

from .core import compute_pvc, is_in_pvc
from .epsilon import (
    compute_critical_epsilon,
    compute_epsilon_pvc,
    is_in_epsilon_pvc,
)

__version__ = "0.1.0"

__all__ = [
    # Core PVC functions
    "compute_pvc",
    "is_in_pvc",
    # Epsilon-PVC functions
    "compute_epsilon_pvc",
    "is_in_epsilon_pvc",
    "compute_critical_epsilon",
    # Version
    "__version__",
]

