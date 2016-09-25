import interleaving as il
import numpy as np
np.random.seed(0)


class TestProbabilisticMultileave(object):
    n = 512     # Number of times of probabilistic tests
    nn = n * n  # Number of times of more probabilistic tests
    pm = il.Probabilistic()

    def _assert_almost_equal(self, a, b, places=2):
        assert round(a, places) == round(b, places)

    def test_sanity(self):
        rankings = [[0]]
        assert self.pm.multileave(*rankings) == [0]

    def test_uniform(self):
        rankings = [[0], [1], [2]]
        l = len(rankings)
        ideal = 1.0 / l
        counts = [0.0] * l
        for i in range(0, self.nn):
            counts[self.pm.multileave(*rankings)[0]] += 1
        for j in range(0, l):
            self._assert_almost_equal(ideal, counts[j] / self.nn)

    def test_round_robin(self):
        rankings = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        for i in range(0, self.n):
            result = self.pm.multileave(*rankings)
            result.sort()
            assert result == [0, 1, 2]

    def test_softmax(self):
        rankings = [[0, 1, 2]]
        ideals = {0: 0.86, 1: 0.11, 2: 0.03}
        counts = {}
        for d in ideals:
            counts[d] = 0.0
        for i in range(0, self.nn):
            counts[self.pm.multileave(*rankings)[0]] += 1
        for d in rankings[0]:
            self._assert_almost_equal(ideals[d], counts[d] / self.nn)

    def test_interaction(self):
        rankings = [[0, 1], [1, 2]]
        ideals = {0: 0.44, 1: 0.50, 2: 0.06}
        counts = {}
        for d in ideals:
            counts[d] = 0.0
        for i in range(0, self.nn):
            counts[self.pm.multileave(*rankings)[0]] += 1
        for d in ideals:
            self._assert_almost_equal(ideals[d], counts[d] / self.nn)

    def test_uniqueness(self):
        rankings = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ]
        for i in range(0, self.n):
            ranking = self.pm.multileave(*rankings)
            ranking.sort()
            uniq_ranking = list(set(ranking))
            uniq_ranking.sort()
            assert ranking == uniq_ranking

    def test_no_shortage(self):
        rankings = [[0], [0, 1], [0, 1, 2]]
        assert 1 == len(self.pm.multileave(*rankings))
