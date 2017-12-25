import interleaving as il
from interleaving import ProbabilisticRanking
from interleaving import TeamRanking
import json
import numpy as np
from collections import defaultdict
from .test_methods import TestMethods

class TestProbabilistic(TestMethods):
    def test_score_interleave(self):
        ranking = ProbabilisticRanking([[1, 2], [2, 3]], [1, 2])
        result = il.Probabilistic.compute_scores(ranking, [0, 1])
        assert result.allocations == {
            (0, 0): ([2, 0], 1 / (1 + 0.125) * (0.125 / (0.125))),
            (0, 1): ([1, 1], 1 / (1 + 0.125) * (1 / (1 + 0.125))),
            (1, 0): ([1, 1], 0.0),
            (1, 1): ([0, 2], 0.0),
        }

    def test_evaluate_interleave(self):
        ranking = ProbabilisticRanking(
            [[1, 2, 3, 4], [2, 3, 4, 1]],
            [1, 2, 3, 4])
        self.evaluate(il.Probabilistic, ranking, [1, 2], [(1, 0)])

    def test_score_multileave(self):
        ranking = ProbabilisticRanking([[1, 2], [2, 1], [2, 3]], [1, 2])
        result = il.Probabilistic.compute_scores(ranking, [0, 1])
        ideal = {
            (0, 0): (
                [2, 0, 0],
                np.exp(
                    np.log(1.0 / (2 * (1.0 + 0.125))) +
                    np.log(0.125 / (2 * (0.125)))
                )
            ),
            (0, 1): ([1, 1, 0],
                np.exp(
                    np.log(1.0 / (2 * (1.0 + 0.125))) +
                    np.log(1.0 / (2 * (1.0)))
                )
            ),
            (0, 2): ([1, 0, 1],
                np.exp(
                    np.log(1.0 / (2 * (1.0 + 0.125))) +
                    np.log(1.0 / (2 * (1.0 + 0.125)))
                )
            ),
            (1, 0): ([1, 1, 0],
                np.exp(
                    np.log(0.125 / (2 * (1.0 + 0.125))) +
                    np.log(0.125 / (2 * (0.125)))
                )
            ),
            (1, 1): ([0, 2, 0],
                np.exp(
                    np.log(0.125 / (2 * (1.0 + 0.125))) +
                    np.log(1.0 / (2 * (1.0)))
                )
            ),
            (1, 2): ([0, 1, 1],
                np.exp(
                    np.log(0.125 / (2 * (1.0 + 0.125))) +
                    np.log(1.0 / (2 * (1.0 + 0.125)))
                )
            ),
        }
        p_sum = np.sum([p for _, p in ideal.values()])
        ideal = {a: (o, p / p_sum) for a, (o, p) in ideal.items()}

        # C++ version does not keep allocations
        if "cProbabilistic" in str(il.Probabilistic.__module__):
            return

        for a in result.allocations:
            assert ideal[a][0] == result.allocations[a][0]
            self.assert_almost_equal(ideal[a][1], result.allocations[a][1])

    def test_evaluate_multileave(self):
        ranking = ProbabilisticRanking([[1, 2], [2, 1], [2, 3]], [1, 2])
        self.evaluate(il.Probabilistic, ranking, [0, 1],
            [(0, 1), (0, 2), (1, 2)])

    def test_init_sampling(self):
        p = il.Probabilistic([[1, 2], [1, 3]], sample_num=20000, replace=False)
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

    def test_dump(self, tmpdir):
        tmpfile = str(tmpdir) + '/probabilistic.json'
        p = il.Probabilistic([[1, 2], [1, 3]], sample_num=10, replace=False)
        p.dump_rankings(tmpfile)
        with open(tmpfile, 'r') as f:
            obj = json.load(f)
        # Test keys
        s = {str(hash(r)) for r in p._rankings}
        assert s == set(obj.keys())
        # Test rankings
        l1 = sorted(p._rankings)
        l2 = sorted([v['ranking']['ranking_list'] for v in obj.values()])
        assert l1 == l2
        # Test lists
        j1 = [r.lists for r in p._rankings]
        j2 = [r['ranking']['lists'] for r in obj.values()]
        assert j1 == j2

    def test_softmax(self):
        softmax = il.Probabilistic.Softmax(2.0, [0, 1, 2])
        p = softmax.delete(0)
        assert p == 1.0 * 1.0 / np.sum([1.0, 1.0 / 4.0, 1.0 / 9.0])
        softmax.reset()
        p = softmax.delete(1)
        assert p == 1.0 / 4.0 * 1.0 / np.sum([1.0, 1.0 / 4.0, 1.0 / 9.0])
        softmax.reset()
        p = softmax.delete(2)
        assert p == 1.0 / 9.0 * 1.0 / np.sum([1.0, 1.0 / 4.0, 1.0 / 9.0])
        softmax.reset()
        p = softmax.delete(0)
        assert p == 1.0 * 1.0 / np.sum([1.0, 1.0 / 4.0, 1.0 / 9.0])
        p = softmax.delete(1)
        self.assert_almost_equal(p,
            1.0 / 4.0 * 1.0 / np.sum([1.0 / 4.0, 1.0 / 9.0]))
        p = softmax.delete(2)
        self.assert_almost_equal(p, 1.0)

    def test_sampling(self):
        n = 5000
        softmax = il.Probabilistic.Softmax(2.0, [0, 1, 2])
        result = defaultdict(int)
        for i in range(n):
            sample = softmax.sample()
            result[sample] += 1
            softmax.reset()

        denominator = np.sum([1.0, 1.0 / 4.0, 1.0 / 9.0])
        self.assert_almost_equal(result[0] / n, 1.0 / denominator)
        self.assert_almost_equal(result[1] / n, 1.0 / 4.0 / denominator)
        self.assert_almost_equal(result[2] / n, 1.0 / 9.0 / denominator)
