import interleaving as il
from interleaving import ProbabilisticRanking
from interleaving import TeamRanking
import json
import numpy as np
from .test_methods import TestMethods
np.random.seed(0)

class TestProbabilistic(TestMethods):
    def test_evaluate_interleave(self):
        ranking = ProbabilisticRanking(
            [[1, 2, 3, 4], [2, 3, 4, 1]],
            [1, 2, 3, 4])
        self.evaluate(il.Probabilistic, ranking, [1, 2], [(1, 0)])

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
        # Test lists
        j1 = [r.lists for r in p._rankings]
        j2 = [r['ranking']['lists'] for r in obj.values()]
        assert j1 == j2

    # TODO: Test evaluation of multileaved result
    # def test_evaluate_multileave(self):
        # ranking = TeamRanking(team_indices=[0, 1], contents=[1, 2, 3, 4])
        # ranking.teams = {0: set([1]), 1: set([2]), 2: set([0])}
        # self.evaluate(il.Probabilistic, ranking, [0, 1, 2], [])
        # self.evaluate(il.Probabilistic, ranking, [0, 2],    [(2, 0), (1, 0)])
        # self.evaluate(il.Probabilistic, ranking, [1, 2],    [(0, 2), (1, 2)])
        # self.evaluate(il.Probabilistic, ranking, [0, 1],    [(0, 1), (2, 1)])
        # self.evaluate(il.Probabilistic, ranking, [0],       [(2, 0), (2, 1)])
        # self.evaluate(il.Probabilistic, ranking, [1],       [(0, 1), (0, 2)])
        # self.evaluate(il.Probabilistic, ranking, [2],       [(1, 0), (1, 2)])
        # self.evaluate(il.Probabilistic, ranking, [],        [])
