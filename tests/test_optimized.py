import interleaving as il
import numpy as np
import pytest
np.random.seed(0)
from .test_methods import TestMethods

class TestOptimized(TestMethods):

    def test_raise_value_error(self):
        with pytest.raises(ValueError):
            # None for `sample_num` is not allowed
            il.Optimized([[0], [1]])

    def test_init_sampling(self):
        b = il.Optimized([[1, 2], [2, 3]], sample_num=100)
        b.dump_rankings('./tmp_om.json')
        samples = set([tuple(b.interleave()) for i in range(1000)])
        assert len(samples) == 3
        assert (1, 2) in samples
        assert (2, 1) in samples
        assert (2, 3) in samples

    def test__unbiasedness_constraints(self):
        lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        b = il.Optimized(lists, sample_num=100)
        res = b._unbiasedness_constraints(lists, b._rankings)
        assert res.shape[0] == (3-1)*3
        assert res.shape[1] == len(b._rankings)

        lists = [[1, 2], [2, 3]]
        b = il.Optimized(lists, sample_num=100)
        res = b._unbiasedness_constraints(lists, b._rankings)
        ideal = {
            (1, 2): [1-1.0/3, 1.5-1.0-1.0/3],
            (2, 3): [0.5-1.0, 0.5+1.0/3-1.5],
            (2, 1): [0.5-1.0, 1.5-1.0-1.0/3]}
        assert res.shape == (2, 3)
        for i in range(2):
            for j, r in enumerate(b._rankings):
                res[i, j] = ideal[tuple(r)][i]

        lists = [[1, 2], [1, 3], [1, 4]]
        b = il.Optimized(lists, sample_num=100)
        rankings = []
        r = il.Ranking([1, 2])
        r.credits = {
            0: {1: 1.0, 2: 0.5}, 1: {1: 1.0, 2: 1.0/3}, 2: {1: 1.0, 2: 1.0/3}}
        rankings.append(r)
        r = il.Ranking([1, 3])
        r.credits = {
            0: {1: 1.0, 3: 1.0/3}, 1: {1: 1.0, 3: 0.5}, 2: {1: 1.0, 3: 1.0/3}}
        rankings.append(r)
        r = il.Ranking([1, 4])
        r.credits = {
            0: {1: 1.0, 4: 1.0/3}, 1: {1: 1.0, 4: 1.0/3}, 2: {1: 1.0, 4: 0.5}}
        rankings.append(r)
        res = b._unbiasedness_constraints(lists, rankings)
        assert res.shape == (4, 3)
        assert res[0, 0] == 0 # Rank 1, Ranking 1, System 1 - System 2
        assert res[0, 1] == 0 # Rank 1, Ranking 2, System 1 - System 2
        assert res[0, 2] == 0 # Rank 1, Ranking 3, System 1 - System 2
        # Rank 2, Ranking 1, System 1 - System 2
        self.assert_almost_equal(res[1, 0], 0.5-1.0/3)
        # Rank 2, Ranking 2, System 1 - System 2
        self.assert_almost_equal(res[1, 1], 1.0/3-0.5)
        assert res[1, 2] == 0 # Rank 2, Ranking 3, System 1 - System 2
        assert res[2, 0] == 0 # Rank 1, Ranking 1, System 2 - System 3
        assert res[2, 1] == 0 # Rank 1, Ranking 2, System 2 - System 3
        assert res[2, 2] == 0 # Rank 1, Ranking 3, System 2 - System 3
        assert res[3, 0] == 0 # Rank 2, Ranking 1, System 2 - System 3
        # Rank 2, Ranking 2, System 2 - System 3
        self.assert_almost_equal(res[3, 1], 0.5-1.0/3)
        # Rank 2, Ranking 3, System 2 - System 3
        self.assert_almost_equal(res[3, 2], 1.0/3-0.5)

    def test__sensitivity(self):
        lists = [[1, 2], [2, 3]]
        b = il.Optimized(lists, sample_num=100)
        rankings = []
        r = il.Ranking([1, 2])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 1.0}}
        rankings.append(r)
        r = il.Ranking([2, 1])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 1.0}}
        rankings.append(r)
        r = il.Ranking([2, 3])
        r.credits = {0: {2: 0.5, 3: 1.0/3}, 1: {2: 1.0, 3: 0.5}}
        rankings.append(r)
        res = b._sensitivity(lists, rankings)
        assert len(res) == 3
        assert res[0] == ((1 + 0.5 * 0.5 - (1.75+1.0/3)/2) ** 2 + (1.0/3+0.5 - (1.75+1.0/3)/2) ** 2)
        assert res[1] == ((0.5 + 0.5 - (2.0+0.5/3)/2) ** 2 + (1.0+1.0/3*0.5 - (2.0+0.5/3)/2) ** 2)
        assert res[2] == ((0.5 + 0.5 * 1.0/3 - (1.75+0.5/3)/2) ** 2\
            + (1.0 + 0.5 * 0.5 - (1.75+0.5/3)/2) ** 2)

    def test__compute_probabilities(self):
        lists = [[1, 2], [2, 3]]
        b = il.Optimized(lists, sample_num=100)
        rankings = []
        r = il.Ranking([1, 2])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 1.0}}
        rankings.append(r)
        r = il.Ranking([2, 1])
        r.credits = {0: {1: 1.0, 2: 0.5}, 1: {1: 1.0/3, 2: 1.0}}
        rankings.append(r)
        r = il.Ranking([2, 3])
        r.credits = {0: {2: 0.5, 3: 1.0/3}, 1: {2: 1.0, 3: 0.5}}
        rankings.append(r)
        is_success, p, minimum = b._compute_probabilities(lists, rankings)
        assert is_success
        assert (p >= 0).all()
        assert (p <= 1).all()
        assert minimum >= 0
        self.assert_almost_equal(np.sum(p), 1)
        self.assert_almost_equal(np.inner([1-1.0/3, -0.5, -0.5], p), 0)
        self.assert_almost_equal(np.inner([0.5-1.0/3, 0.5-1.0/3, -1+1.0/3], p), 0)
        self.assert_almost_equal(p[0], 0.4285714273469387)
        self.assert_almost_equal(p[1], 0.37142857025306114)
        self.assert_almost_equal(p[2], 0.20000000240000002)

    def test_interleave(self):
        lists = [[1, 2], [2, 3]]
        b = il.Optimized(lists, sample_num=100)
        rankings, probabilities = zip(*b.ranking_distribution)
        assert set([(1, 2), (2, 1), (2, 3)]) == set([tuple(r) for r in rankings])
        ideals = {
            (1, 2): 0.4285714273469387,
            (2, 1): 0.37142857025306114,
            (2, 3): 0.20000000240000002
            }
        for i in range(len(probabilities)):
            r = rankings[i]
            self.assert_almost_equal(
                probabilities[i], ideals[tuple(r)], error_rate=0.01)

        trials = 200000
        counts = {(1, 2): 0, (2, 1): 0, (2, 3): 0}
        for i in range(trials):
            r = b.interleave()
            counts[tuple(r)] += 1
        for r, c in counts.items():
            self.assert_almost_equal(
                float(c)/trials, ideals[tuple(r)], error_rate=0.01)

    def test_evaluate(self):
        lists = [[1, 2], [2, 3]]
        b = il.Optimized(lists, sample_num=100)
        samples = [b.interleave() for i in range(1000)]
        rankings = {tuple(r): r for r in samples}

        ideals = {
            (1, 2): {
                0: [(0, 1)],
                1: [(0, 1)],
                2: [(1, 0)],
                3: []},
            (2, 1): {
                0: [(0, 1)],
                1: [(1, 0)],
                2: [(0, 1)],
                3: []},
            (2, 3): {
                0: [(1, 0)],
                1: [(1, 0)],
                2: [(1, 0)],
                3: []},
        }
        for r in rankings:
            self.evaluate(il.Optimized, rankings[r], [0, 1], set(ideals[r][0]))
            self.evaluate(il.Optimized, rankings[r], [0], set(ideals[r][1]))
            self.evaluate(il.Optimized, rankings[r], [1], set(ideals[r][2]))
            self.evaluate(il.Optimized, rankings[r], [], set(ideals[r][3]))
