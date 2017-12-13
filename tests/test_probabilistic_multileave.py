from collections import defaultdict
import interleaving as il
import numpy as np
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

    def test_selected_ranker(self):
        rankings = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        pm = il.Probabilistic(rankings)
        selected_rankers = defaultdict(int)
        for i in range(self.n):
            r = pm.interleave()
            selected_rankers[r[0] % 3] += 1
        self.assert_almost_equal(selected_rankers[0] / self.n, 1 / 3)
        self.assert_almost_equal(selected_rankers[1] / self.n, 1 / 3)
        self.assert_almost_equal(selected_rankers[2] / self.n, 1 / 3)

        selected_rankers = defaultdict(int)
        for i in range(self.n):
            pm = il.Probabilistic(rankings)
            r = pm.interleave()
            selected_rankers[r[0] % 3] += 1
        self.assert_almost_equal(selected_rankers[0] / self.n, 1 / 3)
        self.assert_almost_equal(selected_rankers[1] / self.n, 1 / 3)
        self.assert_almost_equal(selected_rankers[2] / self.n, 1 / 3)

    def test_compute_scores(self):
        rankings = [[0, 1, 2], [2, 1, 0], [1, 0, 2]]
        pm = il.Probabilistic(rankings)
        prefs = defaultdict(int)
        credits = defaultdict(float)
        for i in range(self.n):
            r = pm.interleave()
            clicks = []
            for idx, d in enumerate(r):
                if np.random.rand() < (d + 0.1) / 3.3:
                    clicks = [idx]
            res = pm.compute_scores(r, clicks)
            for idx in res:
                credits[idx] += res[idx]
        for i in range(len(credits)):
            for j in range(i+1, len(credits)):
                if credits[i] > credits[j]:
                    prefs[(i, j)] += 1
                elif credits[j] > credits[i]:
                    prefs[(j, i)] += 1
        print(credits)
        print(prefs)
        assert prefs[(1, 0)] > prefs[(0, 1)]
        assert prefs[(1, 2)] > prefs[(2, 1)]
        assert prefs[(2, 0)] > prefs[(0, 2)]
