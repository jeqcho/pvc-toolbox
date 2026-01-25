"""Epsilon-Proportional Veto Core (ε-PVC) computation functions.

This module implements the continuous relaxation of the Proportional Veto Core,
where a coalition T can block alternative a if:
    |T|/n ≥ 1 - |B|/m + ε
where B is the set of alternatives that all members of T strictly prefer to a.

References
----------
Based on the biclique reduction framework from:
Egor Ianovski, Aleksei Y. Kondratev (2023). "Computing the proportional veto core".
Extended here to handle the continuous ε-PVC case.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from ._flow import FlowNetwork
from .core import _validate_preferences


def _compute_flow_and_critical_epsilon(
    alternative: str,
    profile: List[List[str]],
    candidates: List[str],
    n: int,
    m: int,
) -> float:
    """
    Compute the critical epsilon for an alternative using max-flow.

    Returns the critical epsilon ε* such that:
    - Alternative is ε-blocked when ε ≤ ε*
    - Alternative is in ε-PVC when ε > ε*
    """
    # Build flow network with scaling factors r=m, t=n
    # Node indexing:
    #   0              : source S
    #   1..n           : voter nodes
    #   n+1..n+(m-1)   : candidate nodes (excluding alternative)
    #   n+(m-1)+1      : sink T
    sink_index = 1 + n + (m - 1)
    network = FlowNetwork(sink_index + 1)

    S = 0
    T = sink_index

    # Precompute positions for each voter
    pos: List[Dict[str, int]] = []
    for voter_ranking in profile:
        pos.append({c: i for i, c in enumerate(voter_ranking)})

    # Add S -> voter edges with capacity r = m
    for vi in range(n):
        network.add_edge(S, 1 + vi, m)

    # Map candidates (except alternative) to node IDs
    cand_to_node: Dict[str, int] = {}
    node_cursor = 1 + n
    for d in candidates:
        if d == alternative:
            continue
        cand_to_node[d] = node_cursor
        network.add_edge(node_cursor, T, n)
        node_cursor += 1

    # For each voter, connect to candidates ranked WORSE than alternative
    INF = n * m
    for vi in range(n):
        v_node = 1 + vi
        rank_alt = pos[vi][alternative]
        worse_tail = profile[vi][rank_alt + 1 :]
        for d in worse_tail:
            d_node = cand_to_node[d]
            network.add_edge(v_node, d_node, INF)

    # Compute max flow
    F = network.max_flow(S, T)

    # Compute critical epsilon
    total_vertices = 2 * m * n - n
    S_a = total_vertices - F
    epsilon_star = (S_a / (m * n)) - 1.0

    return epsilon_star


def compute_epsilon_pvc(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
    epsilon: float,
) -> List[str]:
    """
    Compute the Epsilon-Proportional Veto Core (ε-PVC) for a preference profile.

    The ε-PVC is the set of alternatives that cannot be blocked by any coalition
    under the relaxed blocking condition: |T|/n ≥ 1 - |B|/m + ε.

    Parameters
    ----------
    preferences
        Matrix where preferences[rank][voter] is the alternative at that rank
        for that voter. Row 0 is most preferred.
    alternatives
        List of all alternative names.
    epsilon
        The epsilon relaxation parameter. Must satisfy 0 ≤ ε < 1.
        - ε = 0: equivalent to the continuous (non-rounded) PVC
        - ε > 0: relaxes the blocking condition, expanding the core
        - Larger ε makes blocking harder, so more alternatives survive

    Returns
    -------
    List[str]
        List of alternatives in the ε-PVC.

    Raises
    ------
    ValueError
        If epsilon is negative or >= 1, or if the profile is invalid.

    Examples
    --------
    >>> preferences = [
    ...     ["a", "b", "c"],
    ...     ["b", "c", "a"],
    ...     ["c", "a", "b"],
    ... ]
    >>> alternatives = ["a", "b", "c"]
    >>> compute_epsilon_pvc(preferences, alternatives, epsilon=0.1)
    ['a', 'b', 'c']
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if epsilon >= 1:
        raise ValueError(f"epsilon must be < 1, got {epsilon}")

    profile, candidates, n, m = _validate_preferences(preferences, alternatives)

    # Trivial case: single alternative
    if m == 1:
        return [candidates[0]]

    core: List[str] = []
    for c in candidates:
        eps_star = _compute_flow_and_critical_epsilon(c, profile, candidates, n, m)
        # Alternative is in ε-PVC when ε > ε*
        if epsilon > eps_star:
            core.append(c)

    return core


def is_in_epsilon_pvc(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
    alternative: str,
    epsilon: float,
) -> bool:
    """
    Check if a specific alternative is in the Epsilon-Proportional Veto Core.

    Parameters
    ----------
    preferences
        Matrix where preferences[rank][voter] is the alternative at that rank.
    alternatives
        List of all alternative names.
    alternative
        The alternative to check.
    epsilon
        The epsilon relaxation parameter. Must satisfy 0 ≤ ε < 1.

    Returns
    -------
    bool
        True if the alternative is in the ε-PVC, False otherwise.

    Examples
    --------
    >>> preferences = [
    ...     ["a", "b"],
    ...     ["b", "a"],
    ... ]
    >>> alternatives = ["a", "b"]
    >>> is_in_epsilon_pvc(preferences, alternatives, "a", epsilon=0.1)
    True
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if epsilon >= 1:
        raise ValueError(f"epsilon must be < 1, got {epsilon}")

    if alternative not in alternatives:
        raise ValueError(f"Alternative '{alternative}' is not in the alternatives list")

    profile, candidates, n, m = _validate_preferences(preferences, alternatives)

    # Trivial case: single alternative
    if m == 1:
        return True

    eps_star = _compute_flow_and_critical_epsilon(
        alternative, profile, candidates, n, m
    )
    return epsilon > eps_star


def compute_critical_epsilon(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
    alternative: str,
) -> float:
    """
    Compute the critical epsilon ε* for a specific alternative.

    The critical epsilon ε* is the threshold value such that:
    - Alternative is ε-blocked (not in ε-PVC) when ε ≤ ε*
    - Alternative is safe (in ε-PVC) when ε > ε*

    Parameters
    ----------
    preferences
        Matrix where preferences[rank][voter] is the alternative at that rank.
    alternatives
        List of all alternative names.
    alternative
        The alternative for which to compute the critical epsilon.

    Returns
    -------
    float
        The critical epsilon ε* for the given alternative.
        - If ε* = 0: alternative is at the boundary (in ε-PVC for any ε > 0)
        - If ε* > 0: alternative requires ε > ε* to be in the ε-PVC
        - If ε* = -1: trivial case (single alternative, always in core)

    Examples
    --------
    >>> preferences = [
    ...     ["a", "b", "c"],
    ...     ["b", "c", "a"],
    ...     ["c", "a", "b"],
    ... ]
    >>> alternatives = ["a", "b", "c"]
    >>> eps = compute_critical_epsilon(preferences, alternatives, "a")
    >>> # If eps = 0.05, then 'a' is in ε-PVC for any ε > 0.05
    """
    if alternative not in alternatives:
        raise ValueError(f"Alternative '{alternative}' is not in the alternatives list")

    profile, candidates, n, m = _validate_preferences(preferences, alternatives)

    # Trivial case: single alternative is never blocked
    if m == 1:
        return -1.0

    return _compute_flow_and_critical_epsilon(alternative, profile, candidates, n, m)

