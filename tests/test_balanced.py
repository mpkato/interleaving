import interleaving as il
import numpy as np
np.random.seed(0)
from .test_methods import TestMethods

class TestBalanced(TestMethods):

    def test_interleave(self):
        self.interleave(il.Balanced, [1, 2], [2, 3], [(1, 2), (2, 1, 3)])
        self.interleave(il.Balanced, [1, 2], [3, 4], [(1, 3, 2), (3, 1, 4)])

    def test_evaluate(self):
        ranking = il.Ranking([1, 2])
        ranking.a = [1, 2]
        ranking.b = [2, 3]
        self.evaluate(il.Balanced, ranking, [0, 1], (0, 0))
        self.evaluate(il.Balanced, ranking, [0], (1, 0))
        self.evaluate(il.Balanced, ranking, [1], (0, 1))
        self.evaluate(il.Balanced, ranking, [], (0, 0))

        ranking = il.Ranking([2, 1, 3])
        ranking.a = [1, 2]
        ranking.b = [2, 3]
        self.evaluate(il.Balanced, ranking, [0, 1, 2], (0, 0))
        self.evaluate(il.Balanced, ranking, [0, 1], (0, 0))
        self.evaluate(il.Balanced, ranking, [0, 2], (0, 1))
        self.evaluate(il.Balanced, ranking, [1, 2], (0, 0))
        self.evaluate(il.Balanced, ranking, [0], (0, 1))
        self.evaluate(il.Balanced, ranking, [1], (1, 0))
        self.evaluate(il.Balanced, ranking, [2], (0, 1))
        self.evaluate(il.Balanced, ranking, [], (0, 0))
