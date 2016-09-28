import interleaving as il
import numpy as np
np.random.seed(0)
from .test_methods import TestMethods

class TestTeamDraft(TestMethods):

    def test_interleave(self):
        self.interleave(il.TeamDraft, 2, [1, 2], [2, 3], [(1, 2), (2, 1)])
        self.interleave(il.TeamDraft, 3, [1, 2], [2, 3], [(1, 2, 3), (2, 1, 3)])
        self.interleave(il.TeamDraft, 4, [1, 2], [2, 3], [(1, 2, 3), (2, 1, 3)])
        self.interleave(il.TeamDraft, 2, [1, 2], [3, 4], [(1, 3), (3, 1)])
        self.interleave(il.TeamDraft, 3, [1, 2], [3, 4],
            [(1, 3, 2), (1, 3, 4), (3, 1, 2), (3, 1, 4)])

        # check teams
        td = il.TeamDraft()
        res = td.interleave(2, [1, 2], [2, 3])
        assert set(res.teams[0]) == set([1])
        assert set(res.teams[1]) == set([2])

        res = td.interleave(2, [1, 2], [3, 4])
        assert set(res.teams[0]) == set([1])
        assert set(res.teams[1]) == set([3])

    def test_multileave(self):
        self.multileave(il.TeamDraft, 2, [1, 2], [2, 3], [(1, 2), (2, 1)])
        self.multileave(il.TeamDraft, 2, [1, 2], [3, 4], [(1, 3), (3, 1)])

    def test_evaluate(self):
        ranking = il.Ranking([1, 2])
        ranking.teams = {}
        ranking.teams[0] = [1]
        ranking.teams[1] = [2]
        self.evaluate(il.TeamDraft, ranking, [0, 1], [])
        self.evaluate(il.TeamDraft, ranking, [0], [(0, 1)])
        self.evaluate(il.TeamDraft, ranking, [1], [(1, 0)])
        self.evaluate(il.TeamDraft, ranking, [], [])

        ranking = il.Ranking([1, 3, 4])
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

