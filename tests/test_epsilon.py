"""Tests for epsilon-PVC computation and critical epsilon values.

Adapted from test_epsilon_pvc.py for the new pvc_toolbox API
with transposed matrix input format.
"""

from __future__ import annotations

import math

import pytest

from pvc_toolbox import (
    compute_critical_epsilon,
    compute_epsilon_pvc,
    is_in_epsilon_pvc,
)


class TestCriticalEpsilon:
    """Tests for compute_critical_epsilon function."""

    def test_specific_profile(self) -> None:
        """
        Test critical epsilon computation on a 3-voter, 4-alternative profile.

        Profile (columns = voters):
            | c | d | a |
            | b | a | b |
            | a | b | d |
            | d | c | c |

        Voter 0: c > b > a > d
        Voter 1: d > a > b > c
        Voter 2: a > b > d > c

        Expected critical epsilon values:
            - a: ε* = 0.0
            - b: ε* = 0.0
            - c: ε* ≈ 0.416666... (= 5/12)
            - d: ε* ≈ 0.166666... (= 1/6)
        """
        preferences = [
            ["c", "d", "a"],  # rank 0
            ["b", "a", "b"],  # rank 1
            ["a", "b", "d"],  # rank 2
            ["d", "c", "c"],  # rank 3
        ]
        alternatives = ["a", "b", "c", "d"]

        eps_a = compute_critical_epsilon(preferences, alternatives, "a")
        eps_b = compute_critical_epsilon(preferences, alternatives, "b")
        eps_c = compute_critical_epsilon(preferences, alternatives, "c")
        eps_d = compute_critical_epsilon(preferences, alternatives, "d")

        # a and b should be in standard PVC (ε* = 0)
        assert math.isclose(eps_a, 0.0, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(eps_b, 0.0, rel_tol=1e-9, abs_tol=1e-9)

        # c should have ε* = 5/12 ≈ 0.416666...
        expected_eps_c = 5.0 / 12.0
        assert math.isclose(eps_c, expected_eps_c, rel_tol=1e-9, abs_tol=1e-9)

        # d should have ε* = 1/6 ≈ 0.166666...
        expected_eps_d = 1.0 / 6.0
        assert math.isclose(eps_d, expected_eps_d, rel_tol=1e-9, abs_tol=1e-9)

    def test_single_alternative(self) -> None:
        """Test that a single alternative has critical epsilon = -1."""
        preferences = [["a", "a", "a"]]
        alternatives = ["a"]

        eps = compute_critical_epsilon(preferences, alternatives, "a")
        assert eps == -1.0

    def test_unanimous_top_preference(self) -> None:
        """Test that unanimously top-ranked alternative has ε* = 0."""
        preferences = [
            ["a", "a", "a"],
            ["b", "c", "b"],
            ["c", "b", "c"],
        ]
        alternatives = ["a", "b", "c"]

        eps_a = compute_critical_epsilon(preferences, alternatives, "a")
        assert math.isclose(eps_a, 0.0, rel_tol=1e-9, abs_tol=1e-9)

    def test_critical_epsilon_range(self) -> None:
        """Test that critical epsilon values are in valid range."""
        preferences = [
            ["c", "d", "a"],
            ["b", "a", "b"],
            ["a", "b", "d"],
            ["d", "c", "c"],
        ]
        alternatives = ["a", "b", "c", "d"]

        for alt in alternatives:
            eps = compute_critical_epsilon(preferences, alternatives, alt)
            assert math.isfinite(eps)
            assert eps >= -1.0
            assert eps < 1.0

    def test_invalid_alternative(self) -> None:
        """Test that checking non-existent alternative raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="not in the alternatives list"):
            compute_critical_epsilon(preferences, alternatives, "z")

    def test_ten_voter_profile(self) -> None:
        """
        Test critical epsilon on 10-voter, 5-alternative profile.

        Expected: b has ε* = 0.2, c has ε* = 0.3
        """
        preferences = [
            ["a", "a", "e", "d", "c", "d", "d", "a", "b", "a"],  # rank 0
            ["e", "c", "c", "e", "d", "a", "a", "e", "e", "b"],  # rank 1
            ["b", "e", "a", "a", "a", "b", "e", "d", "c", "d"],  # rank 2
            ["d", "d", "d", "c", "e", "e", "b", "b", "a", "e"],  # rank 3
            ["c", "b", "b", "b", "b", "c", "c", "c", "d", "c"],  # rank 4
        ]
        alternatives = ["a", "b", "c", "d", "e"]

        eps_b = compute_critical_epsilon(preferences, alternatives, "b")
        eps_c = compute_critical_epsilon(preferences, alternatives, "c")

        assert math.isclose(eps_b, 0.2, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(eps_c, 0.3, rel_tol=1e-9, abs_tol=1e-9)

    def test_ten_voter_profile_2(self) -> None:
        """
        Test critical epsilon on another 10-voter, 5-alternative profile.

        Expected: e has ε* = 0.2, c has ε* = 0.3
        """
        preferences = [
            ["d", "e", "d", "e", "d", "e", "d", "a", "b", "e"],  # rank 0
            ["c", "a", "b", "a", "b", "b", "b", "b", "d", "a"],  # rank 1
            ["e", "d", "a", "b", "a", "c", "a", "d", "a", "d"],  # rank 2
            ["b", "b", "c", "d", "e", "a", "c", "c", "c", "c"],  # rank 3
            ["a", "c", "e", "c", "c", "d", "e", "e", "e", "b"],  # rank 4
        ]
        alternatives = ["a", "b", "c", "d", "e"]

        eps_e = compute_critical_epsilon(preferences, alternatives, "e")
        eps_c = compute_critical_epsilon(preferences, alternatives, "c")

        assert math.isclose(eps_e, 0.2, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(eps_c, 0.3, rel_tol=1e-9, abs_tol=1e-9)


class TestComputeEpsilonPVC:
    """Tests for compute_epsilon_pvc function."""

    def test_membership_below_critical(self) -> None:
        """Test ε-PVC membership when ε is below critical values."""
        preferences = [
            ["c", "d", "a"],
            ["b", "a", "b"],
            ["a", "b", "d"],
            ["d", "c", "c"],
        ]
        alternatives = ["a", "b", "c", "d"]

        # ε = 0.1 is below d's critical epsilon (~0.1667)
        result = compute_epsilon_pvc(preferences, alternatives, 0.1)
        assert "d" not in result
        assert "c" not in result

    def test_membership_between_critical_values(self) -> None:
        """Test ε-PVC membership when ε is between critical values."""
        preferences = [
            ["c", "d", "a"],
            ["b", "a", "b"],
            ["a", "b", "d"],
            ["d", "c", "c"],
        ]
        alternatives = ["a", "b", "c", "d"]

        # ε = 0.2 is above d's ε* (~0.1667) but below c's ε* (~0.4167)
        result = compute_epsilon_pvc(preferences, alternatives, 0.2)
        assert "d" in result
        assert "c" not in result

    def test_membership_above_critical(self) -> None:
        """Test ε-PVC membership when ε is above critical values."""
        preferences = [
            ["c", "d", "a"],
            ["b", "a", "b"],
            ["a", "b", "d"],
            ["d", "c", "c"],
        ]
        alternatives = ["a", "b", "c", "d"]

        # ε = 0.5 is above both d's and c's critical epsilon
        result = compute_epsilon_pvc(preferences, alternatives, 0.5)
        assert "d" in result
        assert "c" in result

    def test_single_alternative(self) -> None:
        """Test that single alternative is in ε-PVC for any valid ε."""
        preferences = [["a"]]
        alternatives = ["a"]

        result = compute_epsilon_pvc(preferences, alternatives, 0.5)
        assert result == ["a"]

    def test_invalid_epsilon_negative(self) -> None:
        """Test that negative epsilon raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="non-negative"):
            compute_epsilon_pvc(preferences, alternatives, -0.1)

    def test_invalid_epsilon_too_large(self) -> None:
        """Test that epsilon >= 1 raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="< 1"):
            compute_epsilon_pvc(preferences, alternatives, 1.0)


class TestIsInEpsilonPVC:
    """Tests for is_in_epsilon_pvc function."""

    def test_basic_membership(self) -> None:
        """Test basic ε-PVC membership check."""
        preferences = [
            ["c", "d", "a"],
            ["b", "a", "b"],
            ["a", "b", "d"],
            ["d", "c", "c"],
        ]
        alternatives = ["a", "b", "c", "d"]

        # a and b should be in ε-PVC for small ε (they have ε* = 0)
        assert is_in_epsilon_pvc(preferences, alternatives, "a", 0.01)
        assert is_in_epsilon_pvc(preferences, alternatives, "b", 0.01)

        # c should not be in ε-PVC for ε = 0.3 (its ε* ≈ 0.4167)
        assert not is_in_epsilon_pvc(preferences, alternatives, "c", 0.3)

        # c should be in ε-PVC for ε = 0.5
        assert is_in_epsilon_pvc(preferences, alternatives, "c", 0.5)

    def test_at_critical_value(self) -> None:
        """Test membership exactly at critical epsilon boundary."""
        preferences = [
            ["c", "d", "a"],
            ["b", "a", "b"],
            ["a", "b", "d"],
            ["d", "c", "c"],
        ]
        alternatives = ["a", "b", "c", "d"]

        eps_c = compute_critical_epsilon(preferences, alternatives, "c")

        # At exactly ε*, should be blocked
        assert not is_in_epsilon_pvc(preferences, alternatives, "c", eps_c)

        # Just above ε*, should be in core
        assert is_in_epsilon_pvc(preferences, alternatives, "c", eps_c + 1e-6)

    def test_consistency_with_compute(self) -> None:
        """Test that is_in_epsilon_pvc is consistent with compute_epsilon_pvc."""
        preferences = [
            ["a", "b", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
        ]
        alternatives = ["a", "b", "c"]
        epsilon = 0.3

        eps_pvc = compute_epsilon_pvc(preferences, alternatives, epsilon)

        for alt in alternatives:
            expected = alt in eps_pvc
            actual = is_in_epsilon_pvc(preferences, alternatives, alt, epsilon)
            assert actual == expected

    def test_invalid_alternative(self) -> None:
        """Test that checking non-existent alternative raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="not in the alternatives list"):
            is_in_epsilon_pvc(preferences, alternatives, "z", 0.1)

    def test_invalid_epsilon(self) -> None:
        """Test that invalid epsilon values raise errors."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError):
            is_in_epsilon_pvc(preferences, alternatives, "a", -0.1)

        with pytest.raises(ValueError):
            is_in_epsilon_pvc(preferences, alternatives, "a", 1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

