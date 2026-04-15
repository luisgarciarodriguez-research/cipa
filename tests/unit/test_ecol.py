"""Unit tests for ECoL measures: F3, N1, N2, L1."""

import numpy as np
import pytest

from cipa.ecol import compute_f3, compute_l1, compute_n1, compute_n2


def two_gaussians(n0=50, n1=50, sep=5.0, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=0.0, scale=0.5, size=(n0, d))
    X1 = rng.normal(loc=sep, scale=0.5, size=(n1, d))
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


def overlapping(n=100, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------

def test_f3_separable_is_zero():
    X, y = two_gaussians(sep=10.0)
    assert compute_f3(X, y) == 0.0


def test_f3_overlapping_is_high():
    X, y = overlapping()
    assert compute_f3(X, y) > 0.5


def test_f3_in_range():
    X, y = two_gaussians(sep=1.0)
    assert 0.0 <= compute_f3(X, y) <= 1.0


def test_f3_single_feature():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    y = np.array([0, 0, 1, 1])
    assert compute_f3(X, y) == 0.0


# ---------------------------------------------------------------------------
# N1
# ---------------------------------------------------------------------------

def test_n1_separable_is_near_zero():
    # MST always has ≥1 inter-class edge; for well-separated data it's exactly 1 → n1 = 2/N
    X, y = two_gaussians(sep=10.0)
    n1, subsampled = compute_n1(X, y)
    assert n1 < 0.05  # only the 2 bridge points
    assert not subsampled


def test_n1_overlapping_is_high():
    X, y = overlapping()
    n1, _ = compute_n1(X, y)
    assert n1 > 0.3


def test_n1_in_range():
    X, y = two_gaussians(sep=1.0)
    n1, _ = compute_n1(X, y)
    assert 0.0 <= n1 <= 1.0


def test_n1_subsample_path():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.array([1] * 40 + [0] * 160)
    n1, subsampled = compute_n1(X, y, max_exact=100, subsample_size=50, random_state=0)
    assert subsampled
    assert 0.0 <= n1 <= 1.0


# ---------------------------------------------------------------------------
# N2
# ---------------------------------------------------------------------------

def test_n2_separable_is_low():
    X, y = two_gaussians(sep=10.0)
    n2, _ = compute_n2(X, y)
    # Separated: intra << inter → N2_raw small → N2norm = N2_raw/(1+N2_raw) small
    assert n2 < 0.15


def test_n2_overlapping_is_high():
    X, y = overlapping()
    n2, _ = compute_n2(X, y)
    assert n2 > 0.3


def test_n2_in_range():
    X, y = two_gaussians(sep=2.0)
    n2, _ = compute_n2(X, y)
    assert 0.0 <= n2 <= 1.0


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------

def test_l1_linearly_separable_is_low():
    X, y = two_gaussians(sep=10.0)
    l1, converged = compute_l1(X, y, random_state=0)
    assert converged
    assert l1 < 0.1


def test_l1_overlapping_is_higher():
    X, y = overlapping()
    l1, _ = compute_l1(X, y, random_state=0)
    assert l1 > 0.2


def test_l1_in_range():
    X, y = two_gaussians(sep=1.0)
    l1, converged = compute_l1(X, y, random_state=0)
    assert 0.0 <= l1 <= 1.0
