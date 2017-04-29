import interleaving as il
from interleaving import CreditRanking
import json
import numpy as np
import pytest
np.random.seed(0)
from .test_methods import TestMethods

class TestRoughlyOptimized(TestMethods):

    def test__compute_probabilities(self):
        lists = [[1, 2], [2, 3]]
        b = il.RoughlyOptimized(lists, sample_num=3)
        rankings = []
        r = CreditRanking(num_rankers=len(lists), contents=[1, 2])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 1.0}}
        rankings.append(r)
        r = CreditRanking(num_rankers=len(lists), contents=[2, 1])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 1.0}}
        rankings.append(r)
        r = CreditRanking(num_rankers=len(lists), contents=[2, 3])
        r.credits = {0: {2: 0.5, 3: 1.0/3}, 1: {2: 1.0, 3: 0.5}}
        rankings.append(r)
        is_success, p, minimum = b._compute_probabilities_loosely(lists, rankings)
        assert is_success
        assert (p >= 0).all()
        assert (p <= 1).all()
        assert minimum >= 0
        self.assert_almost_equal(np.sum(p), 1)
        self.assert_almost_equal(np.inner([1-1.0/3, -0.5, -0.5], p), 0)
        self.assert_almost_equal(np.inner([0.5-1.0/3, 0.5-1.0/3, -1+1.0/3], p), 0)
        self.assert_almost_equal(p[0], 0.4285714273469387)
        self.assert_almost_equal(p[1], 0.37142857025306114)
        self.assert_almost_equal(p[2], 0.20000000240000002)

    def test__compute_probabilities_biased(self):
        lists = [[1, 2], [2, 3]]
        b = il.RoughlyOptimized(lists, sample_num=3)
        rankings = []
        r = CreditRanking(num_rankers=len(lists), contents=[1, 2])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 0.0}}
        rankings.append(r)
        r = CreditRanking(num_rankers=len(lists), contents=[2, 1])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 0.5, 2: 1.0/3}}
        rankings.append(r)
        r = CreditRanking(num_rankers=len(lists), contents=[2, 3])
        r.credits = {0: {2: 0.5, 3: 1.0/3}, 1: {2: 1.0/3, 3: 0.0}}
        rankings.append(r)
        is_success, p, minimum = b._compute_probabilities_loosely(lists, rankings)
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
        _, _, minimum = b._compute_probabilities_loosely(lists, rankings, 10.0)
        self.assert_almost_equal(
            minimum,
            10.0 * np.sum(b._lambdas) + np.inner(p, b._sigmas),
        )

    def test_interleave(self):
        lists = [[1, 2], [2, 3]]
        b = il.RoughlyOptimized(lists, sample_num=3)
        rankings, probabilities = zip(*b.ranking_distribution)
        assert set([(1, 2), (2, 1), (2, 3)]) == set([tuple(r) for r in rankings])
        ideals = {
            (1, 2): 0.4285714273469387,
            (2, 1): 0.37142857025306114,
            (2, 3): 0.20000000240000002
            }
        for i in range(len(probabilities)):
            r = rankings[i]
            self.assert_almost_equal(
                probabilities[i], ideals[tuple(r)], error_rate=0.01)

        trials = 200000
        counts = {(1, 2): 0, (2, 1): 0, (2, 3): 0}
        for i in range(trials):
            r = b.interleave()
            counts[tuple(r)] += 1
        for r, c in counts.items():
            self.assert_almost_equal(
                float(c)/trials, ideals[tuple(r)], error_rate=0.01)
