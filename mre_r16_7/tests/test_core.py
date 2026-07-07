import unittest
import numpy as np

from toe_mre.core import (
    CAConfig,
    random_pair,
    forward_pair,
    backward_pair,
    evolve,
    rewind,
    exhaustive_permutation_check,
    reversibility_trials,
)
from toe_mre.geometry import spectral_dimension_summary


class TestR167Core(unittest.TestCase):
    def test_single_step_inverse(self):
        prev, curr = random_pair(8, seed=123)
        a, b = forward_pair(prev, curr)
        prev2, curr2 = backward_pair(a, b)
        self.assertTrue(np.array_equal(prev, prev2))
        self.assertTrue(np.array_equal(curr, curr2))

    def test_many_step_inverse(self):
        prev, curr = random_pair(12, seed=7)
        a, b = evolve(prev, curr, 25)
        prev2, curr2 = rewind(a, b, 25)
        self.assertTrue(np.array_equal(prev, prev2))
        self.assertTrue(np.array_equal(curr, curr2))

    def test_exhaustive_permutation_small(self):
        result = exhaustive_permutation_check(n=2)
        self.assertTrue(result["is_permutation"])
        self.assertEqual(result["state_count"], 256)
        self.assertEqual(result["image_count"], 256)

    def test_random_trials(self):
        result = reversibility_trials(CAConfig(n=10, steps=30, seeds=5))
        self.assertTrue(result["passed"])

    def test_spectral_dimension_sanity(self):
        result = spectral_dimension_summary(n=24)
        self.assertTrue(result["within_broad_tolerance"])


if __name__ == "__main__":
    unittest.main()