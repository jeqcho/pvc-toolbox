"""Tests for input validation.

This module tests that invalid inputs are properly rejected with
appropriate error messages.
"""

from __future__ import annotations

import pytest

from pvc_toolbox import compute_pvc, is_in_pvc, compute_epsilon_pvc


class TestPreferencesValidation:
    """Tests for preferences matrix validation."""

    def test_empty_alternatives(self) -> None:
        """Test that empty alternatives list raises error."""
        preferences = [["a"]]
        alternatives: list[str] = []

        with pytest.raises(ValueError, match="non-empty"):
            compute_pvc(preferences, alternatives)

    def test_empty_preferences(self) -> None:
        """Test that empty preferences matrix raises error."""
        preferences: list[list[str]] = []
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="non-empty"):
            compute_pvc(preferences, alternatives)

    def test_no_voters(self) -> None:
        """Test that preferences with no voters raises error."""
        preferences = [[], []]  # 2 ranks, 0 voters
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="at least one voter"):
            compute_pvc(preferences, alternatives)

    def test_mismatched_row_lengths(self) -> None:
        """Test that rows with different lengths raise error."""
        preferences = [
            ["a", "b", "c"],  # 3 voters
            ["b", "a"],      # 2 voters
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="voters"):
            compute_pvc(preferences, alternatives)

    def test_wrong_number_of_ranks(self) -> None:
        """Test that wrong number of ranks raises error."""
        preferences = [
            ["a", "b"],  # only 1 rank for 2 alternatives
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="rows"):
            compute_pvc(preferences, alternatives)

    def test_duplicate_alternatives(self) -> None:
        """Test that duplicate alternatives raise error."""
        preferences = [
            ["a", "a"],
            ["b", "b"],
        ]
        alternatives = ["a", "a"]

        with pytest.raises(ValueError, match="duplicates"):
            compute_pvc(preferences, alternatives)

    def test_unknown_alternative_in_preferences(self) -> None:
        """Test that unknown alternative in preferences raises error."""
        preferences = [
            ["a", "b"],
            ["z", "a"],  # 'z' is not in alternatives
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="not in the alternatives list"):
            compute_pvc(preferences, alternatives)

    def test_duplicate_in_voter_ranking(self) -> None:
        """Test that duplicate in voter's ranking raises error."""
        preferences = [
            ["a", "a"],
            ["a", "b"],  # voter 0 has 'a' twice
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="duplicate"):
            compute_pvc(preferences, alternatives)


class TestIsInPVCValidation:
    """Tests for is_in_pvc validation."""

    def test_alternative_not_in_list(self) -> None:
        """Test that checking non-existent alternative raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="not in the alternatives list"):
            is_in_pvc(preferences, alternatives, "c")


class TestEpsilonPVCValidation:
    """Tests for epsilon-PVC validation."""

    def test_negative_epsilon(self) -> None:
        """Test that negative epsilon raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="non-negative"):
            compute_epsilon_pvc(preferences, alternatives, -0.5)

    def test_epsilon_equals_one(self) -> None:
        """Test that epsilon = 1 raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="< 1"):
            compute_epsilon_pvc(preferences, alternatives, 1.0)

    def test_epsilon_greater_than_one(self) -> None:
        """Test that epsilon > 1 raises error."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        with pytest.raises(ValueError, match="< 1"):
            compute_epsilon_pvc(preferences, alternatives, 1.5)


class TestValidInputs:
    """Tests that valid inputs are accepted."""

    def test_valid_two_alternatives(self) -> None:
        """Test valid 2-alternative profile."""
        preferences = [
            ["a", "b"],
            ["b", "a"],
        ]
        alternatives = ["a", "b"]

        result = compute_pvc(preferences, alternatives)
        assert isinstance(result, list)

    def test_valid_three_alternatives(self) -> None:
        """Test valid 3-alternative profile."""
        preferences = [
            ["a", "b", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
        ]
        alternatives = ["a", "b", "c"]

        result = compute_pvc(preferences, alternatives)
        assert isinstance(result, list)

    def test_valid_single_voter(self) -> None:
        """Test valid single-voter profile."""
        preferences = [
            ["a"],
            ["b"],
            ["c"],
        ]
        alternatives = ["a", "b", "c"]

        result = compute_pvc(preferences, alternatives)
        assert isinstance(result, list)

    def test_valid_single_alternative(self) -> None:
        """Test valid single-alternative profile."""
        preferences = [["a", "a", "a"]]
        alternatives = ["a"]

        result = compute_pvc(preferences, alternatives)
        assert result == ["a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

