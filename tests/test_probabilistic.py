import interleaving as il
from interleaving import TeamRanking
import json
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
        # Test teams
        l1 = [r.teams for r in p._rankings]
        l2 = [v['ranking']['teams'] for v in obj.values()]
        assert len(l1) == len(l2)
        for i1 in l1:
            i1 = {str(k): list(v) for k, v in i1.items()}
            assert i1 in l2
        for i2 in l2:
            i2 = {int(k): set(v) for k, v in i2.items()}
            assert i2 in l1

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

