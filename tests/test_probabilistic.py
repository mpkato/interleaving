import interleaving as il
from interleaving import TeamRanking
import numpy as np
from .test_methods import TestMethods
np.random.seed(0)

class TestProbabilistic(TestMethods):
    def test_evaluate_interleave(self):
        ranking = TeamRanking([0, 1])
        ranking += [10, 20]
        ranking.teams = {0: set([10]), 1: set([20])}
        self.evaluate(il.Probabilistic, ranking, [0, 1], [])
        self.evaluate(il.Probabilistic, ranking, [0],    [(0, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],    [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [],     [])

        ranking = TeamRanking([0, 1])
        ranking += [2, 1, 3]
        ranking.teams = {0: set([2]), 1: set([1, 3])}
        self.evaluate(il.Probabilistic, ranking, [0, 1, 2], [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [0, 2],    [])
        self.evaluate(il.Probabilistic, ranking, [1, 2],    [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [0, 1],    [])
        self.evaluate(il.Probabilistic, ranking, [0],       [(0, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],       [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [2],       [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [],        [])

    def test_init_sampling(self):
        p = il.Probabilistic([[1, 2], [1, 3]], sample_num=200000, replace=False)
        rankings, probabilities = zip(*p.ranking_distribution)
        ideal = set([(1, 3), (1, 2), (2, 1), (2, 3), (3, 1), (3, 2)])
        assert ideal == set([tuple(r) for r in rankings])
        ideal_prob = {
            (1, 2): 0.444444444, (1, 3): 0.444444444,
            (2, 1): 0.049382716, (2, 3): 0.00617284,
            (3, 1): 0.049382716, (3, 2): 0.00617284
        }
        for ranking, prob in zip(rankings, probabilities):
            self.assert_almost_equal(prob, ideal_prob[tuple(ranking)])

        res = p.interleave()
        assert tuple(res) in ideal

    def test_evaluate_multileave(self):
        ranking = TeamRanking([0, 1, 2])
        ranking += [0, 1, 2]
        ranking.teams = {0: set([1]), 1: set([2]), 2: set([0])}
        self.evaluate(il.Probabilistic, ranking, [0, 1, 2], [])
        self.evaluate(il.Probabilistic, ranking, [0, 2],    [(2, 0), (1, 0)])
        self.evaluate(il.Probabilistic, ranking, [1, 2],    [(0, 2), (1, 2)])
        self.evaluate(il.Probabilistic, ranking, [0, 1],    [(0, 1), (2, 1)])
        self.evaluate(il.Probabilistic, ranking, [0],       [(2, 0), (2, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],       [(0, 1), (0, 2)])
        self.evaluate(il.Probabilistic, ranking, [2],       [(1, 0), (1, 2)])
        self.evaluate(il.Probabilistic, ranking, [],        [])

