import interleaving as il
from interleaving import PairwisePreferenceRanking
import json
from .test_methods import TestMethods

class TestPairwisePreference(TestMethods):

    def test_interleave(self):
        #               Method                 Rankings          Max length
        #               Possible interleaved rankings
        self.interleave(il.PairwisePreference, [[1, 2], [2, 3]], 2,
                        [(1, 2), (2, 1), (1, 3), (2, 3)])
        self.interleave(il.PairwisePreference, [[1, 2], [2, 3]], 3, 
                        [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1)])
        self.interleave(il.PairwisePreference, [[1, 2], [2, 3]], 4,
                        [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1)])
        self.interleave(il.PairwisePreference, [[1, 2], [3, 4]], 2,
                        [(1, 2), (1, 3), (1, 4), (3, 1), (3, 2), (3, 4)])
        self.interleave(il.PairwisePreference, [[1, 2], [3, 4]], 3,
            [
                (1, 2, 3), (1, 2, 4), (1, 3, 2), (1, 3, 4), (1, 4, 2), (1, 4, 3), 
                (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 2, 4), (3, 4, 1), (3, 4, 2),
            ])

        # check lists
        pp = il.PairwisePreference([[1, 2], [2, 3]])
        res = pp.interleave()
        assert tuple(res.lists[0]) == tuple([1, 2])
        assert tuple(res.lists[1]) == tuple([2, 3])


    def test_pairwise_preference_ranking(self):
        pp = il.PairwisePreference([[1, 2, 3], [2, 3, 1]], sample_num=100)
        rankings, _ = zip(*pp.ranking_distribution)

        assert len(rankings) == 4
        assert set(map(tuple, rankings))\
            == set(map(tuple, [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1]]))

    def test_dump(self, tmpdir):
        tmpfile = str(tmpdir) + '/team_draft.json'
        pp = il.PairwisePreference([[1, 2, 3], [2, 3, 1]], sample_num=100)
        pp.dump_rankings(tmpfile)
        with open(tmpfile, 'r') as f:
            obj = json.load(f)
        # Test keys
        s = {str(hash(r)) for r in pp._rankings}
        assert s == set(obj.keys())
        # Test rankings
        l1 = sorted(pp._rankings)
        l2 = sorted([v['ranking']['ranking_list'] for v in obj.values()])
        assert l1 == l2
        # Test lists
        l1 = [r.lists for r in pp._rankings]
        l2 = [v['ranking']['lists'] for v in obj.values()]
        assert l1 == l2

    def test_multileave(self):
        self.interleave(il.PairwisePreference, [[1, 2], [2, 3], [3, 4]], 2,
            [
                        (1, 2), (1, 3), (1, 4), 
                (2, 1),         (2, 3), (2, 4), 
                (3, 1), (3, 2),         (3, 4),
            ])
        self.interleave(il.PairwisePreference, [[1, 2], [3, 4], [5, 6]], 2,
            [
                        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
                (3, 1), (3, 2),         (3, 4), (3, 5), (3, 6), 
                (5, 1), (5, 2), (5, 3), (5, 4),         (5, 6), 
            ])

    def test_evaluate(self):
        pass
