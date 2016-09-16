import interleaving as il
import numpy as np
np.random.seed(0)

class TestMethods(object):

    def interleave(self, method, a, b, ideals, num=100):
        results = []
        for i in range(num):
            res = method().interleave(a, b)
            results.append(tuple(res))
        results = set(results)
        possible_results = set([tuple(i) for i in ideals])
        assert results == possible_results

    def evaluate(self, method, ranking, clicks, result):
        res = method().evaluate(ranking, clicks)
        assert res == result

