import interleaving as il
import numpy as np
import pytest

class TestMethods(object):

    @pytest.yield_fixture(autouse=True)
    def fix_seed(self):
        np.random.seed(0)

    def assert_almost_equal(self, a, b, error_rate=0.01):
        assert abs(a-b) < error_rate

    def interleave(self, method, lists, k, ideals, num=100):
        results = []
        for i in range(num):
            res = method(lists, max_length=k).interleave()
            results.append(tuple(res))
        results = set(results)
        possible_results = set([tuple(i) for i in ideals])
        assert results == possible_results

    def evaluate(self, method, ranking, clicks, result):
        res = method.evaluate(ranking, clicks)
        assert set(res) == set(result)

