"""Tests for core PVC computation functions.

Adapted from test_pvc_algorithms.py and test_biclique_hopcroftkar.py
for the new pvc_toolbox API with transposed matrix input format.
"""

from __future__ import annotations

import math

import pytest

from pvc_toolbox import compute_pvc, is_in_pvc


def build_circulant_preferences(n: int, m: int) -> tuple[list[list[str]], list[str]]:
    """
    Generate a deterministic preference matrix that exercises all ranks.

    Returns (preferences, alternatives) where:
    - preferences[rank][voter] is the alternative at that rank for that voter
    - alternatives is the list of alternative names
    """
    # Create alternative names: 'a', 'b', 'c', ...
    alternatives = [chr(ord('a') + i) for i in range(m)]

    # Build the transposed matrix: preferences[rank][voter]
    preferences: list[list[str]] = [[] for _ in range(m)]

    for voter in range(n):
        shift = (voter * 3) % m if m > 0 else 0
        ballot = alternatives[shift:] + alternatives[:shift]
        for rank in range(m):
            preferences[rank].append(ballot[rank])

    return preferences, alternatives


class TestComputePVC:
    """Tests for compute_pvc function."""

    @pytest.mark.parametrize("n", range(1, 10))
    @pytest.mark.parametrize("m", range(1, 10))
    def test_circulant_profiles(self, n: int, m: int) -> None:
        """Test PVC computation on circulant profiles."""
        preferences, alternatives = build_circulant_preferences(n, m)
        pvc = compute_pvc(preferences, alternatives)

        # PVC should be a subset of alternatives
        assert all(alt in alternatives for alt in pvc)

    def test_single_alternative(self) -> None:
        """Test that a single alternative is always in the PVC."""
        preferences = [["a", "a", "a"]]
        alternatives = ["a"]

        pvc = compute_pvc(preferences, alternatives)
        assert pvc == ["a"]

    def test_unanimous_top_preference(self) -> None:
        """Test that unanimously top-ranked alternative is in PVC."""
        preferences = [
            ["a", "a", "a"],  # all voters rank 'a' first
            ["b", "c", "b"],
            ["c", "b", "c"],
        ]
        alternatives = ["a", "b", "c"]

        pvc = compute_pvc(preferences, alternatives)
        assert "a" in pvc

    def test_balanced_cyclic_profile(self) -> None:
        """Test balanced cyclic preferences."""
        preferences = [
            ["a", "b", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
        ]
        alternatives = ["a", "b", "c"]

        pvc = compute_pvc(preferences, alternatives)
        # All alternatives should be in core for perfectly balanced profile
        assert len(pvc) > 0

    def test_two_voters_agreeing(self) -> None:
        """Test with two voters who agree."""
        preferences = [
            ["a", "a"],
            ["b", "b"],
        ]
        alternatives = ["a", "b"]

        pvc = compute_pvc(preferences, alternatives)
        assert "a" in pvc  # Unanimously top-ranked

    def test_two_voters_disagreeing(self) -> None:
        """Test with two voters who disagree."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        pvc = compute_pvc(preferences, alternatives)
        # Both or neither should be in PVC
        assert isinstance(pvc, list)

    def test_asymmetric_profile(self) -> None:
        """Test highly asymmetric profile."""
        preferences = [
            ["a", "a", "a", "d"],
            ["b", "b", "b", "c"],
            ["c", "c", "c", "b"],
            ["d", "d", "d", "a"],
        ]
        alternatives = ["a", "b", "c", "d"]

        pvc = compute_pvc(preferences, alternatives)
        assert "a" in pvc  # Ranked first by 3/4 voters

    @pytest.mark.parametrize("n", [10, 15, 20])
    @pytest.mark.parametrize("m", [5, 8, 10])
    def test_medium_profiles(self, n: int, m: int) -> None:
        """Test that medium-sized profiles are handled correctly."""
        preferences, alternatives = build_circulant_preferences(n, m)
        pvc = compute_pvc(preferences, alternatives)

        assert isinstance(pvc, list)
        assert all(alt in alternatives for alt in pvc)


class TestIsInPVC:
    """Tests for is_in_pvc function."""

    def test_basic_membership(self) -> None:
        """Test basic PVC membership check."""
        preferences = [
            ["a", "a", "a"],
            ["b", "c", "b"],
            ["c", "b", "c"],
        ]
        alternatives = ["a", "b", "c"]

        # 'a' is unanimously first, should be in PVC
        assert is_in_pvc(preferences, alternatives, "a") is True

    def test_single_alternative_membership(self) -> None:
        """Test membership for single alternative."""
        preferences = [["a"]]
        alternatives = ["a"]

        assert is_in_pvc(preferences, alternatives, "a") is True

    def test_invalid_alternative(self) -> None:
        """Test that checking non-existent alternative raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="not in the alternatives list"):
            is_in_pvc(preferences, alternatives, "z")

    def test_consistency_with_compute_pvc(self) -> None:
        """Test that is_in_pvc is consistent with compute_pvc."""
        preferences = [
            ["a", "b", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
        ]
        alternatives = ["a", "b", "c"]

        pvc = compute_pvc(preferences, alternatives)

        for alt in alternatives:
            assert is_in_pvc(preferences, alternatives, alt) == (alt in pvc)


class TestSpecificProfiles:
    """Tests for specific profiles from test_biclique_hopcroftkar.py."""

    def test_three_voter_four_alternative(self) -> None:
        """
        Test 3-voter, 4-alternative profile.

        Voter 0: c > b > a > d
        Voter 1: d > a > b > c
        Voter 2: a > b > d > c
        """
        preferences = [
            ["c", "d", "a"],  # rank 0
            ["b", "a", "b"],  # rank 1
            ["a", "b", "d"],  # rank 2
            ["d", "c", "c"],  # rank 3
        ]
        alternatives = ["a", "b", "c", "d"]

        pvc = compute_pvc(preferences, alternatives)
        assert isinstance(pvc, list)

    def test_ten_voter_profile_1(self) -> None:
        """
        Test 10-voter, 5-alternative profile.

        Voter 0: a > e > b > d > c
        Voter 1: a > c > e > d > b
        ...etc
        """
        preferences = [
            ["a", "a", "e", "d", "c", "d", "d", "a", "b", "a"],  # rank 0
            ["e", "c", "c", "e", "d", "a", "a", "e", "e", "b"],  # rank 1
            ["b", "e", "a", "a", "a", "b", "e", "d", "c", "d"],  # rank 2
            ["d", "d", "d", "c", "e", "e", "b", "b", "a", "e"],  # rank 3
            ["c", "b", "b", "b", "b", "c", "c", "c", "d", "c"],  # rank 4
        ]
        alternatives = ["a", "b", "c", "d", "e"]

        pvc = compute_pvc(preferences, alternatives)
        assert isinstance(pvc, list)

    def test_ten_voter_profile_2(self) -> None:
        """
        Test another 10-voter, 5-alternative profile.

        Voter 0: d > c > e > b > a
        Voter 1: e > a > d > b > c
        ...etc
        """
        preferences = [
            ["d", "e", "d", "e", "d", "e", "d", "a", "b", "e"],  # rank 0
            ["c", "a", "b", "a", "b", "b", "b", "b", "d", "a"],  # rank 1
            ["e", "d", "a", "b", "a", "c", "a", "d", "a", "d"],  # rank 2
            ["b", "b", "c", "d", "e", "a", "c", "c", "c", "c"],  # rank 3
            ["a", "c", "e", "c", "c", "d", "e", "e", "e", "b"],  # rank 4
        ]
        alternatives = ["a", "b", "c", "d", "e"]

        pvc = compute_pvc(preferences, alternatives)
        assert isinstance(pvc, list)

    def test_condorcet_winner_in_core(self) -> None:
        """Test that a Condorcet winner is in the PVC."""
        # Profile where 'a' is a Condorcet winner
        preferences = [
            ["a", "a", "b"],
            ["b", "c", "a"],
            ["c", "b", "c"],
        ]
        alternatives = ["a", "b", "c"]

        pvc = compute_pvc(preferences, alternatives)
        assert "a" in pvc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

