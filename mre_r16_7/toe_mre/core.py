"""
Core reversible finite-information dynamics for TOE R16.7.

The model is intentionally minimal:
- binary states on an N x N torus;
- second-order reversible cellular automaton;
- deterministic local map F;
- exact inverse.

The update acts on a pair (previous, current):
    next = local_rule(current) XOR previous
    T(previous, current) = (current, next)

The inverse is:
    previous = local_rule(current) XOR next
    T^{-1}(current, next) = (previous, current)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CAConfig:
    """Configuration for the reversible cellular automaton."""
    n: int = 8
    steps: int = 32
    seeds: int = 20

    def validate(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.seeds < 1:
            raise ValueError("seeds must be >= 1")


def as_binary_grid(x: Array) -> Array:
    """Return an array reduced to binary values."""
    return np.asarray(x, dtype=np.uint8) & np.uint8(1)


def local_rule(x: Array) -> Array:
    """
    Local deterministic rule F on a 2D torus.

    It uses XOR of self plus four von Neumann neighbors. The rule itself
    does not need to be invertible because the second-order construction
    makes the global pair update invertible.
    """
    x = as_binary_grid(x)
    return (
        x
        ^ np.roll(x, 1, axis=0)
        ^ np.roll(x, -1, axis=0)
        ^ np.roll(x, 1, axis=1)
        ^ np.roll(x, -1, axis=1)
    ).astype(np.uint8)


def forward_pair(previous: Array, current: Array) -> Tuple[Array, Array]:
    """One reversible forward step: (previous, current) -> (current, next)."""
    previous = as_binary_grid(previous)
    current = as_binary_grid(current)
    if previous.shape != current.shape:
        raise ValueError("previous and current must have identical shape")
    nxt = local_rule(current) ^ previous
    return current.copy(), nxt.astype(np.uint8)


def backward_pair(current: Array, nxt: Array) -> Tuple[Array, Array]:
    """One exact inverse step: (current, next) -> (previous, current)."""
    current = as_binary_grid(current)
    nxt = as_binary_grid(nxt)
    if current.shape != nxt.shape:
        raise ValueError("current and next must have identical shape")
    previous = local_rule(current) ^ nxt
    return previous.astype(np.uint8), current.copy()


def evolve(previous: Array, current: Array, steps: int) -> Tuple[Array, Array]:
    """Evolve a pair forward for a number of steps."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    a, b = as_binary_grid(previous), as_binary_grid(current)
    for _ in range(steps):
        a, b = forward_pair(a, b)
    return a, b


def rewind(current_minus_1: Array, current: Array, steps: int) -> Tuple[Array, Array]:
    """Rewind a pair backward for a number of steps."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    a, b = as_binary_grid(current_minus_1), as_binary_grid(current)
    for _ in range(steps):
        a, b = backward_pair(a, b)
    return a, b


def random_pair(n: int, seed: int) -> Tuple[Array, Array]:
    """Generate a reproducible random binary state pair."""
    rng = np.random.default_rng(seed)
    previous = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
    current = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
    return previous, current


def pack_pair(previous: Array, current: Array) -> int:
    """
    Pack a binary pair into an integer.

    Used only for small complete-state enumeration.
    """
    previous = as_binary_grid(previous).ravel()
    current = as_binary_grid(current).ravel()
    bits = np.concatenate([previous, current]).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def unpack_pair(value: int, n: int) -> Tuple[Array, Array]:
    """Inverse of pack_pair for an N x N pair."""
    total_bits = 2 * n * n
    if value < 0 or value >= (1 << total_bits):
        raise ValueError("value outside state space")
    bits = np.zeros(total_bits, dtype=np.uint8)
    for i in range(total_bits - 1, -1, -1):
        bits[i] = value & 1
        value >>= 1
    half = n * n
    previous = bits[:half].reshape(n, n)
    current = bits[half:].reshape(n, n)
    return previous, current


def exhaustive_permutation_check(n: int = 2) -> dict:
    """
    Enumerate the full state space for small n and verify that T is a permutation.

    For n=2, the pair has 8 bits, so there are 256 states.
    """
    if n > 3:
        raise ValueError("exhaustive enumeration is intentionally limited to n <= 3")
    total = 1 << (2 * n * n)
    images = set()
    for value in range(total):
        prev, curr = unpack_pair(value, n)
        img = pack_pair(*forward_pair(prev, curr))
        images.add(img)

    return {
        "n": n,
        "state_count": total,
        "image_count": len(images),
        "is_permutation": len(images) == total,
        "uniform_entropy_bits_before": float(np.log2(total)),
        "uniform_entropy_bits_after": float(np.log2(len(images))) if images else 0.0,
    }


def reversibility_trials(config: CAConfig) -> dict:
    """
    Run random trials: evolve forward and then backward, checking exact recovery.
    """
    config.validate()
    failures = []
    for seed in range(config.seeds):
        prev0, curr0 = random_pair(config.n, seed)
        a, b = evolve(prev0, curr0, config.steps)
        prev_back, curr_back = rewind(a, b, config.steps)

        ok = bool(np.array_equal(prev0, prev_back) and np.array_equal(curr0, curr_back))
        if not ok:
            failures.append(seed)

    return {
        "n": config.n,
        "steps": config.steps,
        "seeds": config.seeds,
        "passed": len(failures) == 0,
        "failures": failures,
    }


def hamming_density(x: Array) -> float:
    """Fraction of active bits in one grid."""
    x = as_binary_grid(x)
    return float(x.mean())


def trajectory_observables(n: int = 16, steps: int = 64, seed: int = 0) -> dict:
    """
    Produce simple reproducible observables for one trajectory.

    Hamming density is not required to be conserved. This is included to avoid
    confusing 'conservation of information' with conservation of every macroscopic
    statistic.
    """
    prev, curr = random_pair(n, seed)
    densities = [hamming_density(curr)]
    a, b = prev, curr
    for _ in range(steps):
        a, b = forward_pair(a, b)
        densities.append(hamming_density(b))

    return {
        "n": n,
        "steps": steps,
        "seed": seed,
        "density_start": densities[0],
        "density_end": densities[-1],
        "density_min": min(densities),
        "density_max": max(densities),
        "density_mean": float(np.mean(densities)),
        "note": "Hamming density is a macroscopic observable and is not generally conserved.",
    }