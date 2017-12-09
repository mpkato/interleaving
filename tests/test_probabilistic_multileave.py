import interleaving as il
from .test_methods import TestMethods

class TestProbabilisticMultileave(TestMethods):
    n = 5000     # Number of times of probabilistic tests

    def test_sanity(self):
        rankings = [[0]]
        pm = il.Probabilistic(rankings)
        assert pm.interleave() == [0]

    def test_uniform(self):
        rankings = [[0], [1], [2]]
        l = len(rankings)
        ideal = 1.0 / l
        counts = [0.0] * l
        pm = il.Probabilistic(rankings)
        for i in range(0, self.n):
            counts[pm.interleave()[0]] += 1
        for j in range(0, l):
            self.assert_almost_equal(ideal, counts[j] / self.n)

    def test_round_robin(self):
        rankings = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        pm = il.Probabilistic(rankings)
        for i in range(0, self.n):
            result = pm.interleave()
            result.sort()
            assert result == [0, 1, 2]

    def test_softmax(self):
        rankings = [[0, 1, 2]]
        ideals = {0: 0.86056, 1: 0.10757, 2: 0.03187}
        counts = {}
        for d in ideals:
            counts[d] = 0.0
        pm = il.Probabilistic(rankings)
        for i in range(0, self.n):
            counts[pm.interleave()[0]] += 1
        for d in ideals:
            self.assert_almost_equal(ideals[d], counts[d] / self.n)

    def test_interaction(self):
        rankings = [[0, 1], [1, 2]]
        ideals = {0: 0.44444, 1: 0.50000, 2: 0.05556}
        counts = {}
        for d in ideals:
            counts[d] = 0.0
        pm = il.Probabilistic(rankings)
        for i in range(0, self.n):
            counts[pm.interleave()[0]] += 1
        for d in ideals:
            self.assert_almost_equal(ideals[d], counts[d] / self.n)

    def test_uniqueness(self):
        rankings = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ]
        pm = il.Probabilistic(rankings, max_length=2)
        for i in range(0, self.n):
            ranking = pm.interleave()
            ranking.sort()
            uniq_ranking = list(set(ranking))
            uniq_ranking.sort()
            assert ranking == uniq_ranking

    def test_no_shortage(self):
        rankings = [[0], [0, 1], [0, 1, 2]]
        pm = il.Probabilistic(rankings)
        assert 1 == len(pm.interleave())

