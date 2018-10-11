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

    def test_find_preferences(self):
        ranking = [0, 1, 2, 3, 4]

        clicks = []
        prefs = il.PairwisePreference._find_preferences(ranking, clicks)
        assert set(prefs) == set()

        clicks = [0]
        prefs = il.PairwisePreference._find_preferences(ranking, clicks)
        assert set(prefs) == set([(0, 1)])

        clicks = [2]
        prefs = il.PairwisePreference._find_preferences(ranking, clicks)
        assert set(prefs) == set([
            (2, 0), (2, 1), (2, 3)
        ])

        clicks = [1, 3]
        prefs = il.PairwisePreference._find_preferences(ranking, clicks)
        assert set(prefs) == set([
            (1, 0), (1, 2),
            (3, 0), (3, 2), (3, 4)
        ])

        clicks = [4]
        prefs = il.PairwisePreference._find_preferences(ranking, clicks)
        assert set(prefs) == set([
            (4, 0), (4, 1), (4, 2), (4, 3)
        ])

    def test_compute_scores(self):
        rankings = [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4, 3],
            [2, 1, 4, 0, 3],
        ]
        multileaved_ranking = [0, 1, 2, 3, 4]
        ranking = PairwisePreferenceRanking(rankings, multileaved_ranking)

        scores = il.PairwisePreference.compute_scores(ranking, [0])
        assert scores[0] == 0.0
        assert scores[1] == 0.0
        assert scores[2] == 0.0

        scores = il.PairwisePreference.compute_scores(ranking, [1])
        # (1, 0), (1, 2)
        w = 1 - 1/2
        assert scores[0] == 1 / w
        assert scores[1] == 1 / w
        assert scores[2] == - 1 / w

        scores = il.PairwisePreference.compute_scores(ranking, [2])
        # (2, 0), (2, 1), (2, 3)
        w20 = 1
        w21 = 1 - 1/2
        assert scores[0] == - w20 - 1 / w21
        assert scores[1] == - w20 - 1 / w21
        assert scores[2] ==   w20 + 1 / w21

        scores = il.PairwisePreference.compute_scores(ranking, [1, 2])
        # (1, 0), (2, 0), (2, 3)
        assert scores[0] == -1
        assert scores[1] == -1
        assert scores[2] ==  1

    def test_find_highest_rank_for_all(self):
        rankings = [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ]
        rank = il.PairwisePreference._find_highest_rank_for_all(rankings, 0, 1)
        assert rank == 1

        rank = il.PairwisePreference._find_highest_rank_for_all(rankings, 1, 2)
        assert rank == 2

        rank = il.PairwisePreference._find_highest_rank_for_all(rankings, 0, 4)
        assert rank == 0

    def test_find_highest_rank_for_any(self):
        rankings = [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ]
        rank = il.PairwisePreference._find_highest_rank_for_any(rankings, 0, 1)
        assert rank == 0

        rank = il.PairwisePreference._find_highest_rank_for_any(rankings, 1, 2)
        assert rank == 1

        rank = il.PairwisePreference._find_highest_rank_for_any(rankings, 0, 4)
        assert rank == 0

    def test_find_highest_rank_for_ranking(self):
        ranking = [0, 1, 2, 3, 4]

        rank = il.PairwisePreference._find_highest_rank_for_ranking(ranking, 0, 1)
        assert rank == 0

        rank = il.PairwisePreference._find_highest_rank_for_ranking(ranking, 1, 2)
        assert rank == 1

        rank = il.PairwisePreference._find_highest_rank_for_ranking(ranking, 2, 4)
        assert rank == 2

    def test_get_rank(self):
        ranking = [0, 1, 2, 3, 4]

        rank = il.PairwisePreference._get_rank(ranking, 2)
        assert rank == 2

        rank = il.PairwisePreference._get_rank(ranking, 4)
        assert rank == 4

        rank = il.PairwisePreference._get_rank(ranking, 6)
        assert rank == 5

    def test_compute_probability(self):
        rankings = [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ]
        sup, inf = 2, 1
        r_above = il.PairwisePreference._find_highest_rank_for_all(rankings, sup, inf)
        assert r_above == 2
        w = il.PairwisePreference._compute_probability(r_above, rankings, sup, inf)
        ideal_w = 1 - 1 / (4 - 1)
        assert w == ideal_w

        rankings = [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4, 3],
            [2, 1, 4, 0, 3],
        ]
        sup, inf = 1, 3
        r_above = il.PairwisePreference._find_highest_rank_for_all(rankings, sup, inf)
        assert r_above == 3
        w = il.PairwisePreference._compute_probability(r_above, rankings, sup, inf)
        ideal_w = (1 - 1 / (3 - 1)) * (1 - 1 / (4 - 2))
        assert w == ideal_w
