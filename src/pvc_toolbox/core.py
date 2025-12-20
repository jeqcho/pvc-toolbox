"""Core Proportional Veto Core (PVC) computation functions.

This module provides the main API for computing the Proportional Veto Core
for a preference profile.

References
----------
Egor Ianovski, Aleksei Y. Kondratev (2023). "Computing the proportional veto core".
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set, Tuple

from ._flow import FlowNetwork


def _validate_preferences(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
) -> Tuple[List[List[str]], List[str], int, int]:
    """
    Validate the input preferences matrix and alternatives list.

    Parameters
    ----------
    preferences
        Matrix where preferences[rank][voter] is the alternative at that rank
        for that voter. Each column must be a permutation of alternatives.
    alternatives
        List of all alternative names.

    Returns
    -------
    (profile, alt_list, n, m)
        - profile: transposed list where profile[voter][rank] = alternative
        - alt_list: list of alternatives
        - n: number of voters
        - m: number of alternatives

    Raises
    ------
    ValueError
        If preferences or alternatives are invalid.
    """
    if not alternatives:
        raise ValueError("alternatives must be non-empty")

    m = len(alternatives)
    alt_set = set(alternatives)
    if len(alt_set) != m:
        raise ValueError("alternatives contains duplicates")

    if not preferences:
        raise ValueError("preferences must be non-empty")

    if len(preferences) != m:
        raise ValueError(
            f"preferences must have {m} rows (one per rank), got {len(preferences)}"
        )

    # Get number of voters from first row
    n = len(preferences[0])
    if n == 0:
        raise ValueError("preferences must have at least one voter (column)")

    # Validate each row has same number of voters
    for rank_idx, row in enumerate(preferences):
        if len(row) != n:
            raise ValueError(
                f"Row {rank_idx} has {len(row)} voters, expected {n}"
            )

    # Transpose to profile[voter][rank] and validate each voter's ranking
    profile: List[List[str]] = []
    for voter in range(n):
        voter_ranking: List[str] = []
        for rank in range(m):
            alt = preferences[rank][voter]
            if alt not in alt_set:
                raise ValueError(
                    f"Alternative '{alt}' at position [{rank}][{voter}] "
                    f"is not in the alternatives list"
                )
            voter_ranking.append(alt)

        # Check for duplicates in voter's ranking
        if len(set(voter_ranking)) != m:
            raise ValueError(f"Voter {voter} has duplicate alternatives in ranking")

        profile.append(voter_ranking)

    return profile, list(alternatives), n, m


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclid: returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


def _choose_r_t(n: int, m: int) -> Tuple[int, int, int]:
    """
    Choose integers (r, t, alpha) s.t. r*n = t*m - alpha, t > alpha*n, r > 0.

    This follows the construction in the paper's proof (Theorem 6).

    Returns
    -------
    (r, t, alpha)
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers")

    alpha = math.gcd(m, n)
    g, x, y = _extended_gcd(n, m)
    assert g == alpha

    # We want r*n = t*m - alpha <=> (-r)*n + t*m = alpha
    r0 = -x
    t0 = y

    # General solution: r = r0 + k*(m/alpha), t = t0 + k*(n/alpha)
    step_r = m // alpha
    step_t = n // alpha

    def ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    k_min_t = ceil_div((alpha * n + 1) - t0, step_t)
    k_min_r = 0 if r0 > 0 else ceil_div(1 - r0, step_r)
    k = max(k_min_t, k_min_r)

    r = r0 + k * step_r
    t = t0 + k * step_t

    if not (r > 0 and t > alpha * n):
        extra = 1 + max(0, (alpha * n + 1 - t) // step_t)
        k += extra
        r = r0 + k * step_r
        t = t0 + k * step_t

    return r, t, alpha


def _is_alternative_blocked(
    alternative: str,
    profile: List[List[str]],
    candidates: List[str],
    n: int,
    m: int,
    r: int,
    t: int,
    alpha: int,
) -> bool:
    """
    Check if an alternative is blocked using max-flow.

    Returns True if the alternative is blocked (not in PVC).
    """
    # Build the flow network for this alternative
    # Node indexing:
    #   0                : source S
    #   1..n             : voter nodes
    #   n+1 .. n+(m-1)   : candidate!=alternative nodes
    #   n+(m-1)+1        : sink T
    sink_index = 1 + n + (m - 1)
    network = FlowNetwork(sink_index + 1)

    S = 0
    T = sink_index

    # Precompute positions for each voter
    pos: List[Dict[str, int]] = []
    for voter_ranking in profile:
        pos.append({c: i for i, c in enumerate(voter_ranking)})

    # Add S -> voter edges (capacity r for each voter)
    for vi in range(n):
        network.add_edge(S, 1 + vi, r)

    # Map candidates (except alternative) to node ids and add candidate -> T edges
    cand_to_node: Dict[str, int] = {}
    node_cursor = 1 + n
    for d in candidates:
        if d == alternative:
            continue
        cand_to_node[d] = node_cursor
        network.add_edge(node_cursor, T, t)
        node_cursor += 1

    # For each voter, connect to candidates ranked WORSE than alternative
    INF = n * r
    for vi in range(n):
        v_node = 1 + vi
        rank_alt = pos[vi][alternative]
        # All candidates after alternative in voter's order are "worse"
        worse_tail = profile[vi][rank_alt + 1 :]
        for d in worse_tail:
            d_node = cand_to_node[d]
            network.add_edge(v_node, d_node, INF)

    # Compute max flow
    F = network.max_flow(S, T)

    # Blocking threshold: F <= t*(m-1) - alpha means blocked
    block_threshold = t * (m - 1) - alpha
    return F <= block_threshold


def compute_pvc(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
) -> List[str]:
    """
    Compute the Proportional Veto Core (PVC) for a preference profile.

    Parameters
    ----------
    preferences
        Matrix where preferences[rank][voter] is the alternative at that rank
        for that voter. Row 0 is the most preferred, row m-1 is least preferred.
        Each column must be a permutation of all alternatives.
    alternatives
        List of all alternative names.

    Returns
    -------
    List[str]
        List of alternatives in the Proportional Veto Core.

    Examples
    --------
    >>> preferences = [
    ...     ["a", "b", "c"],  # rank 0 (most preferred)
    ...     ["b", "c", "a"],  # rank 1
    ...     ["c", "a", "b"],  # rank 2 (least preferred)
    ... ]
    >>> alternatives = ["a", "b", "c"]
    >>> compute_pvc(preferences, alternatives)
    ['a', 'b', 'c']

    Notes
    -----
    Uses a max-flow reduction algorithm from Ianovski & Kondratev (2023).
    If scipy is installed, uses optimized C implementation for max-flow.
    """
    profile, candidates, n, m = _validate_preferences(preferences, alternatives)

    # Trivial case: single alternative
    if m == 1:
        return [candidates[0]]

    # Compute (r, t, alpha) for the profile
    r, t, alpha = _choose_r_t(n, m)

    # Check each candidate
    core: List[str] = []
    for c in candidates:
        if not _is_alternative_blocked(c, profile, candidates, n, m, r, t, alpha):
            core.append(c)

    return core


def is_in_pvc(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
    alternative: str,
) -> bool:
    """
    Check if a specific alternative is in the Proportional Veto Core.

    Parameters
    ----------
    preferences
        Matrix where preferences[rank][voter] is the alternative at that rank
        for that voter.
    alternatives
        List of all alternative names.
    alternative
        The alternative to check.

    Returns
    -------
    bool
        True if the alternative is in the PVC, False otherwise.

    Examples
    --------
    >>> preferences = [
    ...     ["a", "b"],
    ...     ["b", "a"],
    ... ]
    >>> alternatives = ["a", "b"]
    >>> is_in_pvc(preferences, alternatives, "a")
    True
    """
    if alternative not in alternatives:
        raise ValueError(f"Alternative '{alternative}' is not in the alternatives list")

    profile, candidates, n, m = _validate_preferences(preferences, alternatives)

    # Trivial case: single alternative
    if m == 1:
        return True

    # Compute (r, t, alpha) for the profile
    r, t, alpha = _choose_r_t(n, m)

    return not _is_alternative_blocked(
        alternative, profile, candidates, n, m, r, t, alpha
    )

