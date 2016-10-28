import interleaving as il
from interleaving import TeamRanking
import json
import numpy as np
np.random.seed(0)
from .test_methods import TestMethods

class TestTeamDraft(TestMethods):

    def test_interleave(self):
        self.interleave(il.TeamDraft, [[1, 2], [2, 3]], 2, [(1, 2), (2, 1)])
        self.interleave(il.TeamDraft, [[1, 2], [2, 3]], 3, [(1, 2, 3), (2, 1, 3)])
        self.interleave(il.TeamDraft, [[1, 2], [2, 3]], 4, [(1, 2, 3), (2, 1, 3)])
        self.interleave(il.TeamDraft, [[1, 2], [3, 4]], 2, [(1, 3), (3, 1)])
        self.interleave(il.TeamDraft, [[1, 2], [3, 4]], 3,
            [(1, 3, 2), (1, 3, 4), (3, 1, 2), (3, 1, 4)])

        # check teams
        td = il.TeamDraft([[1, 2], [2, 3]])
        res = td.interleave()
        assert set(res.teams[0]) == set([1])
        assert set(res.teams[1]) == set([2])

        td = il.TeamDraft([[1, 2], [3, 4]])
        res = td.interleave()
        assert set(res.teams[0]) == set([1])
        assert set(res.teams[1]) == set([3])

    def test_team_draft_ranking(self):
        td = il.TeamDraft([[1, 2, 3], [2, 3, 1]], sample_num=100)
        rankings, distributions = zip(*td.ranking_distribution)
        assert len(rankings) == 4

    def test_multileave(self):
        self.interleave(il.TeamDraft, [[1, 2], [2, 3], [3, 4]], 2,
            [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)])
        self.interleave(il.TeamDraft, [[1, 2], [3, 4], [5, 6]], 2,
            [(1, 3), (1, 5), (3, 1), (3, 5), (5, 1), (5, 3)])

    def test_evaluate(self):
        ranking = TeamRanking(team_indices=[0, 1], contents=[1, 2])
        ranking.teams = {}
        ranking.teams[0] = [1]
        ranking.teams[1] = [2]
        self.evaluate(il.TeamDraft, ranking, [0, 1], [])
        self.evaluate(il.TeamDraft, ranking, [0], [(0, 1)])
        self.evaluate(il.TeamDraft, ranking, [1], [(1, 0)])
        self.evaluate(il.TeamDraft, ranking, [], [])

        ranking = TeamRanking(team_indices=[0, 1], contents=[1, 3, 4])
        ranking.teams = {}
        ranking.teams[0] = [1]
        ranking.teams[1] = [3, 4]
        self.evaluate(il.TeamDraft, ranking, [0, 1, 2], [(1, 0)])
        self.evaluate(il.TeamDraft, ranking, [0, 1], [])
        self.evaluate(il.TeamDraft, ranking, [0, 2], [])
        self.evaluate(il.TeamDraft, ranking, [1, 2], [(1, 0)])
        self.evaluate(il.TeamDraft, ranking, [0], [(0, 1)])
        self.evaluate(il.TeamDraft, ranking, [1], [(1, 0)])
        self.evaluate(il.TeamDraft, ranking, [2], [(1, 0)])
        self.evaluate(il.TeamDraft, ranking, [], [])

