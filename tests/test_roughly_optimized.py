from .test_optimized import TestOptimized
import interleaving as il
from interleaving import CreditRanking
from interleaving.optimized import Optimized as StrictlyOptimized
import numpy as np
np.random.seed(0)


class TestRoughlyOptimized(TestOptimized):

    def setup(self):
        self.lists = [[1, 2], [2, 3]]
        self.rankings = []
        r = CreditRanking(num_rankers=len(self.lists), contents=[1, 2])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 0.0}}
        self.rankings.append(r)
        r = CreditRanking(num_rankers=len(self.lists), contents=[2, 1])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 0.5, 2: 1.0/3}}
        self.rankings.append(r)
        r = CreditRanking(num_rankers=len(self.lists), contents=[2, 3])
        r.credits = {0: {2: 0.5, 3: 1.0/3}, 1: {2: 1.0/3, 3: 0.0}}
        self.rankings.append(r)
        il.Optimized = il.RoughlyOptimized  # Trick

    def test__compute_probabilities_fail(self):
        b = StrictlyOptimized(self.lists, sample_num=3)  # NOT roughly
        is_success, p, minimum = b._compute_probabilities(
            self.lists,
            self.rankings,
        )
        assert is_success is False

    def test__compute_probabilities_loosely(self):
        b = il.RoughlyOptimized(self.lists, sample_num=3)
        is_success, p, minimum = b._compute_probabilities(
            self.lists,
            self.rankings,
        )
        assert is_success
        self.assert_almost_equal(p[0], 0.0)
        self.assert_almost_equal(p[1], 0.0)
        self.assert_almost_equal(p[2], 1.0)
        self.assert_almost_equal(b._lambdas[0], 0.5 - 1.0/3)
        self.assert_almost_equal(b._lambdas[1], 0.5 - 1.0/3 + 1.0/3 - 0.0)
        self.assert_almost_equal(
            minimum,
            np.sum(b._lambdas) + np.inner(p, b._sigmas),
        )
        _, _, minimum = b._compute_probabilities_loosely(
            self.lists,
            self.rankings,
            bias_weight=10.0,
        )
        self.assert_almost_equal(
            minimum,
            10.0 * np.sum(b._lambdas) + np.inner(p, b._sigmas),
        )
