# PVC Toolbox

Tools for computing the Proportional Veto Core (PVC) and epsilon-PVC for preference profiles.

## Installation

### Basic Installation

```bash
pip install pvc-toolbox
```

### With scipy for faster computation

```bash
pip install pvc-toolbox[fast]
```

## Quick Start

```python
from pvc_toolbox import compute_pvc, is_in_pvc

# Define preferences as matrix: preferences[rank][voter]
# Each column is a voter's ranking (top to bottom = most to least preferred)
preferences = [
    ["a", "b", "c"],  # rank 0 (most preferred)
    ["b", "c", "a"],  # rank 1
    ["c", "a", "b"],  # rank 2 (least preferred)
]
alternatives = ["a", "b", "c"]

# Compute the full PVC
pvc = compute_pvc(preferences, alternatives)
print(f"PVC: {pvc}")  # ['a', 'b', 'c']

# Check if a specific alternative is in the PVC
print(is_in_pvc(preferences, alternatives, "a"))  # True
```

## API Reference

### Core PVC Functions

#### `compute_pvc(preferences, alternatives) -> List[str]`

Compute the Proportional Veto Core for a preference profile.

**Parameters:**
- `preferences`: Matrix where `preferences[rank][voter]` is the alternative at that rank for that voter
- `alternatives`: List of all alternative names

**Returns:** List of alternatives in the PVC

#### `is_in_pvc(preferences, alternatives, alternative) -> bool`

Check if a specific alternative is in the PVC.

**Parameters:**
- `preferences`: Preference matrix
- `alternatives`: List of all alternatives
- `alternative`: The alternative to check

**Returns:** True if the alternative is in the PVC

### Epsilon-PVC Functions

#### `compute_epsilon_pvc(preferences, alternatives, epsilon) -> List[str]`

Compute the ε-Proportional Veto Core with relaxed blocking condition.

**Parameters:**
- `preferences`: Preference matrix
- `alternatives`: List of all alternatives
- `epsilon`: Relaxation parameter (0 ≤ ε < 1)

**Returns:** List of alternatives in the ε-PVC

#### `is_in_epsilon_pvc(preferences, alternatives, alternative, epsilon) -> bool`

Check if a specific alternative is in the ε-PVC.

#### `compute_critical_epsilon(preferences, alternatives, alternative) -> float`

Compute the critical epsilon ε* for an alternative.

The critical epsilon is the threshold such that:
- Alternative is blocked when ε ≤ ε*
- Alternative is in ε-PVC when ε > ε*

**Example:**
```python
from pvc_toolbox import compute_critical_epsilon, is_in_epsilon_pvc

preferences = [
    ["c", "d", "a"],
    ["b", "a", "b"],
    ["a", "b", "d"],
    ["d", "c", "c"],
]
alternatives = ["a", "b", "c", "d"]

# Compute critical epsilon for alternative 'c'
eps_c = compute_critical_epsilon(preferences, alternatives, "c")
print(f"Critical epsilon for 'c': {eps_c:.4f}")  # ~0.4167

# Check membership at different epsilon values
print(is_in_epsilon_pvc(preferences, alternatives, "c", 0.3))  # False
print(is_in_epsilon_pvc(preferences, alternatives, "c", 0.5))  # True
```

## Input Format

The `preferences` parameter is a transposed matrix where:
- Rows represent ranks (0 = most preferred, m-1 = least preferred)
- Columns represent voters

```
          Voter 0  Voter 1  Voter 2
Rank 0      "a"      "b"      "c"    <- most preferred
Rank 1      "b"      "c"      "a"
Rank 2      "c"      "a"      "b"    <- least preferred
```

Each column must be a complete permutation of all alternatives.

## Performance

If scipy is not installed, a warning will be displayed suggesting to install it for faster computation. The package uses:
- **With scipy**: C-optimized max-flow algorithm
- **Without scipy**: Pure Python Dinic's algorithm

## References

Egor Ianovski, Aleksei Y. Kondratev (2023). "Computing the proportional veto core".

## License

MIT License

