import interleaving as il
from .test_methods import TestMethods


class TestProbabilistic(TestMethods):
    def test_evaluate_interleave(self):
        ranking = il.Ranking([10, 20])
        ranking.teams = {0: set([10]), 1: set([20])}
        self.evaluate(il.Probabilistic, ranking, [0, 1], [])
        self.evaluate(il.Probabilistic, ranking, [0],    [(0, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],    [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [],     [])

        ranking = il.Ranking([2, 1, 3])
        ranking.teams = {0: set([2]), 1: set([1, 3])}
        self.evaluate(il.Probabilistic, ranking, [0, 1, 2], [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [0, 2],    [])
        self.evaluate(il.Probabilistic, ranking, [1, 2],    [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [0, 1],    [])
        self.evaluate(il.Probabilistic, ranking, [0],       [(0, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],       [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [2],       [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [],        [])

    def test_evaluate_multileave(self):
        ranking = il.Ranking([0, 1, 2])
        ranking.teams = {0: set([1]), 1: set([2]), 2: set([0])}
        self.evaluate(il.Probabilistic, ranking, [0, 1, 2], [])
        self.evaluate(il.Probabilistic, ranking, [0, 2],    [(2, 0), (1, 0)])
        self.evaluate(il.Probabilistic, ranking, [1, 2],    [(0, 2), (1, 2)])
        self.evaluate(il.Probabilistic, ranking, [0, 1],    [(0, 1), (2, 1)])
        self.evaluate(il.Probabilistic, ranking, [0],       [(2, 0), (2, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],       [(0, 1), (0, 2)])
        self.evaluate(il.Probabilistic, ranking, [2],       [(1, 0), (1, 2)])
        self.evaluate(il.Probabilistic, ranking, [],        [])
