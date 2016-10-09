import interleaving as il
import numpy as np
from collections import defaultdict
from .test_methods import TestMethods
np.random.seed(0)

class TestProbabilisticInterleave(TestMethods):
    n = 512     # Number of times of probabilistic tests
    nn = n * n  # Number of times of more probabilistic tests

    def test_sanity(self):
        assert il.Probabilistic([[0], [0]]).interleave() == [0]

    def test_uniform(self):
        ideal = 0.5
        counts = [0.0, 0.0]
        pm = il.Probabilistic([[0], [1]])
        for i in range(self.nn):
            r = pm.interleave()
            counts[r[0]] += 1
        for j in [0, 1]:
            self.assert_almost_equal(ideal, counts[j] / self.nn)

    def test_ranking_with_teams(self):
        result = defaultdict(int)
        pm = il.Probabilistic([[1, 2, 3], [2, 3, 1]])
        for i in range(self.nn):
            result[pm.interleave()] += 1
        assert len(result) == 24

    def test_memorylessness(self):
        result = []
        pm = il.Probabilistic([[0, 1], [2, 3]])
        for i in range(self.n):
            result.extend(pm.interleave())
        result = list(set(result))
        result.sort()
        assert result == [0, 1, 2, 3]

    def test_softmax(self):
        ideals = {0: 0.86056, 1: 0.10757, 2: 0.03187}
        counts = {}
        for d in ideals:
            counts[d] = 0.0

        pm = il.Probabilistic([[0, 1, 2], [0, 1, 2]])
        for i in range(self.nn):
            counts[pm.interleave()[0]] += 1
        for d in ideals:
            self.assert_almost_equal(ideals[d], counts[d] / self.nn)

    def test_interaction(self):
        ideals = {0: 0.44444, 1: 0.50000, 2: 0.05556}
        counts = {}
        for d in ideals:
            counts[d] = 0.0

        pm = il.Probabilistic([[0, 1], [1, 2]])
        for i in range(self.nn):
            counts[pm.interleave()[0]] += 1
        for d in ideals:
            self.assert_almost_equal(ideals[d], counts[d] / self.nn)

    def test_uniqueness(self):
        pm = il.Probabilistic([[0, 1, 2], [1, 2, 0]])
        for i in range(self.n):
            ranking = pm.interleave()
            ranking.sort()
            uniq_ranking = list(set(ranking))
            uniq_ranking.sort()
            assert ranking == uniq_ranking

    def test_no_shortage(self):
        rankings = [[0, 1], [0, 1, 2]]
        pm = il.Probabilistic(rankings)
        assert 2 == len(pm.interleave())

