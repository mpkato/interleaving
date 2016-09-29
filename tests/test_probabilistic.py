import interleaving as il
from .test_methods import TestMethods


class TestProbabilistic(TestMethods):
    pm = il.Probabilistic()

    def test_evaluate_interleave(self):
        ranking = il.Ranking([10, 20])
        ranking.number_of_rankers = 2
        ranking.rank_to_ranker_index = [0, 1]
        self.evaluate(il.Probabilistic, ranking, [0, 1], [])
        self.evaluate(il.Probabilistic, ranking, [0],    [(0, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],    [(1, 0)])
        self.evaluate(il.Probabilistic, ranking, [],     [])

        ranking = il.Ranking([2, 1, 3])
        ranking.number_of_rankers = 2
        ranking.rank_to_ranker_index = [0, 1, 1]
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
        ranking.number_of_rankers = 3
        ranking.rank_to_ranker_index = [2, 0, 1]
        self.evaluate(il.Probabilistic, ranking, [0, 1, 2], [])
        self.evaluate(il.Probabilistic, ranking, [0, 2],    [(2, 0), (1, 0)])
        self.evaluate(il.Probabilistic, ranking, [1, 2],    [(0, 2), (1, 2)])
        self.evaluate(il.Probabilistic, ranking, [0, 1],    [(0, 1), (2, 1)])
        self.evaluate(il.Probabilistic, ranking, [0],       [(2, 0), (2, 1)])
        self.evaluate(il.Probabilistic, ranking, [1],       [(0, 1), (0, 2)])
        self.evaluate(il.Probabilistic, ranking, [2],       [(1, 0), (1, 2)])
        self.evaluate(il.Probabilistic, ranking, [],        [])
