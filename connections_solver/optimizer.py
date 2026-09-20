"""Exact branch-and-bound optimization for four groups of four words."""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np

CostMode = Literal["sum", "mean"]


def _group_cost(indices: tuple[int, ...], distances: np.ndarray, mode: CostMode) -> float:
    cost = sum(float(distances[a, b]) for position, a in enumerate(indices) for b in indices[:position])
    return cost / 6 if mode == "mean" else cost


def solve_with_cost(
    words: list[str], distance_matrix: np.ndarray, cost_mode: CostMode = "sum"
) -> tuple[list[list[str]], float]:
    """Find the exact minimum-cost partition using canonical branch-and-bound.

    Groups are built from the lowest remaining index, avoiding duplicate partitions.
    A branch is pruned once its cost cannot beat the best complete partition found.
    """
    if len(words) != 16 or distance_matrix.shape != (16, 16):
        raise ValueError("solve requires 16 words and a 16x16 distance matrix")
    if cost_mode not in ("sum", "mean"):
        raise ValueError("cost_mode must be 'sum' or 'mean'")

    # A greedy incumbent makes pruning effective; replace with a stronger heuristic if needed.
    remaining = tuple(range(16))
    greedy_groups = []
    greedy_cost = 0.0
    while remaining:
        first = remaining[0]
        candidate = min(
            (tuple([first, *combo]) for combo in combinations(remaining[1:], 3)),
            key=lambda group: _group_cost(group, distance_matrix, cost_mode),
        )
        greedy_groups.append(candidate)
        greedy_cost += _group_cost(candidate, distance_matrix, cost_mode)
        remaining = tuple(index for index in remaining if index not in candidate)

    best_cost = greedy_cost
    best_groups = greedy_groups

    def search(left: tuple[int, ...], groups: list[tuple[int, ...]], cost: float) -> None:
        nonlocal best_cost, best_groups
        if not left:
            if cost < best_cost:
                best_cost, best_groups = cost, groups.copy()
            return
        first, *rest = left
        for combo in combinations(rest, 3):
            group = (first, *combo)
            group_cost = _group_cost(group, distance_matrix, cost_mode)
            if cost + group_cost >= best_cost:
                continue
            search(tuple(index for index in rest if index not in combo), groups + [group], cost + group_cost)

    search(tuple(range(16)), [], 0.0)
    return [[words[index] for index in group] for group in best_groups], best_cost


def solve(
    words: list[str], distance_matrix: np.ndarray, cost_mode: CostMode = "sum"
) -> list[list[str]]:
    """Return the minimum-cost four-group partition; use solve_with_cost for its cost."""
    return solve_with_cost(words, distance_matrix, cost_mode)[0]
