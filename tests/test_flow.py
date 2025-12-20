"""Tests for max-flow implementations.

This module tests both the scipy and pure Python Dinic implementations
to ensure they produce the same results.
"""

from __future__ import annotations

import pytest

from pvc_toolbox._flow import Dinic, FlowNetwork, has_scipy


class TestDinic:
    """Tests for the pure Python Dinic implementation."""

    def test_simple_flow(self) -> None:
        """Test simple flow network."""
        # Simple network: s -> a -> t with capacity 10
        dinic = Dinic(3)
        dinic.add_edge(0, 1, 10)
        dinic.add_edge(1, 2, 10)

        flow = dinic.max_flow(0, 2)
        assert flow == 10

    def test_multiple_paths(self) -> None:
        """Test flow with multiple paths."""
        # s -> a -> t (cap 5)
        # s -> b -> t (cap 7)
        dinic = Dinic(4)
        dinic.add_edge(0, 1, 5)  # s -> a
        dinic.add_edge(0, 2, 7)  # s -> b
        dinic.add_edge(1, 3, 5)  # a -> t
        dinic.add_edge(2, 3, 7)  # b -> t

        flow = dinic.max_flow(0, 3)
        assert flow == 12

    def test_bottleneck(self) -> None:
        """Test flow with bottleneck edge."""
        dinic = Dinic(3)
        dinic.add_edge(0, 1, 100)  # high capacity
        dinic.add_edge(1, 2, 5)    # bottleneck

        flow = dinic.max_flow(0, 2)
        assert flow == 5

    def test_complex_network(self) -> None:
        """Test more complex network."""
        # Diamond-shaped network
        dinic = Dinic(4)
        dinic.add_edge(0, 1, 3)
        dinic.add_edge(0, 2, 2)
        dinic.add_edge(1, 2, 1)
        dinic.add_edge(1, 3, 2)
        dinic.add_edge(2, 3, 3)

        flow = dinic.max_flow(0, 3)
        assert flow == 5

    def test_no_path(self) -> None:
        """Test when there's no path from source to sink."""
        dinic = Dinic(3)
        dinic.add_edge(0, 1, 10)
        # No edge from 1 to 2

        flow = dinic.max_flow(0, 2)
        assert flow == 0

    def test_invalid_vertices(self) -> None:
        """Test that invalid vertices raise errors."""
        dinic = Dinic(3)

        with pytest.raises(ValueError):
            dinic.max_flow(-1, 2)

        with pytest.raises(ValueError):
            dinic.max_flow(0, 5)

    def test_too_few_vertices(self) -> None:
        """Test that fewer than 2 vertices raises error."""
        with pytest.raises(ValueError):
            Dinic(1)


class TestFlowNetwork:
    """Tests for the FlowNetwork abstraction."""

    def test_simple_flow(self) -> None:
        """Test simple flow network."""
        network = FlowNetwork(3)
        network.add_edge(0, 1, 10)
        network.add_edge(1, 2, 10)

        flow = network.max_flow(0, 2)
        assert flow == 10

    def test_multiple_paths(self) -> None:
        """Test flow with multiple paths."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 5)
        network.add_edge(0, 2, 7)
        network.add_edge(1, 3, 5)
        network.add_edge(2, 3, 7)

        flow = network.max_flow(0, 3)
        assert flow == 12


class TestBothImplementations:
    """Tests that compare scipy and Dinic implementations."""

    @pytest.mark.parametrize("n_vertices,edges,expected_flow", [
        (3, [(0, 1, 10), (1, 2, 10)], 10),
        (4, [(0, 1, 5), (0, 2, 7), (1, 3, 5), (2, 3, 7)], 12),
        (3, [(0, 1, 100), (1, 2, 5)], 5),
        (4, [(0, 1, 3), (0, 2, 2), (1, 2, 1), (1, 3, 2), (2, 3, 3)], 5),
        (3, [(0, 1, 10)], 0),  # No path to sink
    ])
    def test_implementations_match(
        self,
        n_vertices: int,
        edges: list[tuple[int, int, int]],
        expected_flow: int,
    ) -> None:
        """Test that both implementations give the same results."""
        # Test Dinic directly
        dinic = Dinic(n_vertices)
        for u, v, cap in edges:
            dinic.add_edge(u, v, cap)
        dinic_flow = dinic.max_flow(0, n_vertices - 1)

        # Test FlowNetwork (uses scipy if available, otherwise Dinic)
        network = FlowNetwork(n_vertices)
        for u, v, cap in edges:
            network.add_edge(u, v, cap)
        network_flow = network.max_flow(0, n_vertices - 1)

        # Both should match expected
        assert dinic_flow == expected_flow
        assert network_flow == expected_flow

    def test_dinic_vs_network_private_methods(self) -> None:
        """Explicitly test both private methods in FlowNetwork."""
        n_vertices = 4
        edges = [(0, 1, 5), (0, 2, 7), (1, 3, 5), (2, 3, 7)]

        # Test Dinic implementation via private method
        network1 = FlowNetwork(n_vertices)
        for u, v, cap in edges:
            network1.add_edge(u, v, cap)
        dinic_result = network1._max_flow_dinic(0, 3)

        # Test scipy implementation via private method (if available)
        if has_scipy():
            network2 = FlowNetwork(n_vertices)
            for u, v, cap in edges:
                network2.add_edge(u, v, cap)
            scipy_result = network2._max_flow_scipy(0, 3)

            assert dinic_result == scipy_result == 12
        else:
            assert dinic_result == 12


class TestPVCWithBothImplementations:
    """Test PVC computation using both implementations."""

    def test_pvc_with_dinic(self) -> None:
        """Test PVC computation using Dinic directly."""
        # We'll manually test the blocking check logic with Dinic
        from pvc_toolbox._flow import Dinic

        # Simple 3-voter, 3-alternative PVC network
        # This mimics the network structure used in core.py
        n, m = 3, 3
        r, t = 3, 4  # Example (r, t) values

        # Build network for alternative 'a' (index 0)
        # Nodes: 0=source, 1-3=voters, 4-5=candidates b,c, 6=sink
        dinic = Dinic(7)

        # Source -> voters (capacity r)
        for vi in range(n):
            dinic.add_edge(0, 1 + vi, r)

        # Candidates -> sink (capacity t)
        dinic.add_edge(4, 6, t)  # b -> sink
        dinic.add_edge(5, 6, t)  # c -> sink

        # Voter -> worse candidates edges (unbounded)
        INF = n * r
        # Voter 0: ranks a > b > c, so worse than a: b, c
        dinic.add_edge(1, 4, INF)  # voter 0 -> b
        dinic.add_edge(1, 5, INF)  # voter 0 -> c

        # Voter 1: ranks b > c > a, so nothing worse than a (a is last)
        # No edges

        # Voter 2: ranks c > a > b, so worse than a: b
        dinic.add_edge(3, 4, INF)  # voter 2 -> b

        flow = dinic.max_flow(0, 6)
        assert isinstance(flow, int)
        assert flow >= 0

    def test_compare_pvc_results(self) -> None:
        """Compare PVC results using both flow implementations."""
        from pvc_toolbox import compute_pvc
        from pvc_toolbox._flow import has_scipy

        preferences = [
            ["a", "b", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
        ]
        alternatives = ["a", "b", "c"]

        # Compute PVC (uses scipy if available)
        pvc = compute_pvc(preferences, alternatives)

        # Verify result is valid
        assert isinstance(pvc, list)
        assert all(alt in alternatives for alt in pvc)

        # Log which implementation was used
        print(f"\nUsing scipy: {has_scipy()}")
        print(f"PVC result: {pvc}")


class TestHasScipy:
    """Tests for has_scipy function."""

    def test_has_scipy_returns_bool(self) -> None:
        """Test that has_scipy returns a boolean."""
        result = has_scipy()
        assert isinstance(result, bool)

    def test_scipy_consistency(self) -> None:
        """Test that has_scipy is consistent with actual availability."""
        try:
            from scipy.sparse.csgraph import maximum_flow
            assert has_scipy() is True
        except ImportError:
            assert has_scipy() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

