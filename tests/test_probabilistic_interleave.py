import interleaving as il
import numpy as np
np.random.seed(4294967295)


class TestProbabilisticInterleave(object):
    n = 512     # Number of times of probabilistic tests
    nn = n * n  # Number of times of more probabilistic tests
    pm = il.Probabilistic()

    def _assert_almost_equal(self, a, b, places=2):
        assert round(a, places) == round(b, places)

    def test_sanity(self):
        assert self.pm.interleave([0], [0]) == [0]

    def test_uniform(self):
        ideal = 0.5
        counts = [0.0, 0.0]
        for i in range(0, self.nn):
            counts[self.pm.interleave([0], [1])[0]] += 1
        for j in [0, 1]:
            self._assert_almost_equal(ideal, counts[j] / self.nn)

    def test_memorylessness(self):
        result = []
        for i in range(0, self.n):
            result.extend(self.pm.interleave([0, 1], [2, 3]))
        result = list(set(result))
        result.sort()
        assert result == [0, 1, 2, 3]

    def test_softmax(self):
        ideals = {0: 0.86, 1: 0.11, 2: 0.03}
        counts = {}
        for d in ideals:
            counts[d] = 0.0
        for i in range(0, self.nn):
            counts[self.pm.interleave([0, 1, 2], [0, 1, 2])[0]] += 1
        for d in ideals:
            self._assert_almost_equal(ideals[d], counts[d] / self.nn)

    def test_interaction(self):
        ideals = {0: 0.44, 1: 0.50, 2: 0.06}
        counts = {}
        for d in ideals:
            counts[d] = 0.0
        for i in range(0, self.nn):
            counts[self.pm.interleave([0, 1], [1, 2])[0]] += 1
        for d in ideals:
            self._assert_almost_equal(ideals[d], counts[d] / self.nn)

    def test_uniqueness(self):
        for i in range(0, self.n):
            ranking = self.pm.interleave([0, 1, 2], [1, 2, 0])
            ranking.sort()
            uniq_ranking = list(set(ranking))
            uniq_ranking.sort()
            assert ranking == uniq_ranking

    def test_no_shortage(self):
        rankings = [[0, 1], [0, 1, 2]]
        assert 2 == len(self.pm.interleave(*rankings))
