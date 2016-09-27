import interleaving as il
import numpy as np
np.random.seed(0)
from .test_methods import TestMethods

class TestTeamDraft(TestMethods):

    def test_interleave(self):
        self.interleave(il.TeamDraft, [1, 2], [2, 3], [(1, 2), (2, 1)])
        self.interleave(il.TeamDraft, [1, 2], [3, 4],
            [(1, 3, 4), (1, 3, 2), (3, 1, 2), (3, 1, 4)])

        # check teams
        td = il.TeamDraft()
        res = td.interleave([1, 2], [2, 3])
        assert set(res.team_a) == set([1])
        assert set(res.team_b) == set([2])

        res = td.interleave([1, 2], [3, 4])
        if res[-1] == 4:
            assert set(res.team_a) == set([1])
            assert set(res.team_b) == set([3, 4])
        else:
            assert set(res.team_a) == set([1, 2])
            assert set(res.team_b) == set([3])

    def test_multileave(self):
        self.multileave(il.TeamDraft, [1, 2], [2, 3], [(1, 2), (2, 1)])
        self.multileave(il.TeamDraft, [1, 2], [3, 4], [(1, 3), (3, 1)])

    def test_evaluate(self):
        ranking = il.Ranking([1, 2])
        ranking.team_a = [1]
        ranking.team_b = [2]
        self.evaluate(il.TeamDraft, ranking, [0, 1], (0, 0))
        self.evaluate(il.TeamDraft, ranking, [0], (1, 0))
        self.evaluate(il.TeamDraft, ranking, [1], (0, 1))
        self.evaluate(il.TeamDraft, ranking, [], (0, 0))

        ranking = il.Ranking([1, 3, 4])
        ranking.team_a = [1]
        ranking.team_b = [3, 4]
        self.evaluate(il.TeamDraft, ranking, [0, 1, 2], (0, 1))
        self.evaluate(il.TeamDraft, ranking, [0, 1], (0, 0))
        self.evaluate(il.TeamDraft, ranking, [0, 2], (0, 0))
        self.evaluate(il.TeamDraft, ranking, [1, 2], (0, 1))
        self.evaluate(il.TeamDraft, ranking, [0], (1, 0))
        self.evaluate(il.TeamDraft, ranking, [1], (0, 1))
        self.evaluate(il.TeamDraft, ranking, [2], (0, 1))
        self.evaluate(il.TeamDraft, ranking, [], (0, 0))

