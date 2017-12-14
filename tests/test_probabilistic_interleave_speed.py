import interleaving as il
from .test_methods import TestMethods

class TestProbabilisticInterleaveSpeed(TestMethods):

    def test_interleave(self):
        r1 = list(range(100))
        r2 = list(range(50, 150))
        r3 = list(range(100, 200))
        r4 = list(range(150, 250))
        for i in range(100):
            method = il.Probabilistic([r1, r2, r3, r4])
            ranking = method.interleave()
            method.evaluate(ranking, [0, 1, 2])
