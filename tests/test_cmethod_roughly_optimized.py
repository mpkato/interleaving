import interleaving as il
from interleaving.cRoughlyOptimized import RoughlyOptimized
from .test_roughly_optimized import TestRoughlyOptimized
import pytest
import numpy as np

class TestCMethodRoughlyOptimized(TestRoughlyOptimized):
    @pytest.yield_fixture(autouse=True)
    def replace_method(self):
        self.tmp = il.RoughlyOptimized
        il.RoughlyOptimized = RoughlyOptimized
        yield
        il.RoughlyOptimized = self.tmp

    def test_concordance(self):
        # Ensure which variables include which modules
        assert not "cRoughlyOptimized" in self.tmp.__module__
        assert "roughly_optimized" in self.tmp.__module__
        assert "cRoughlyOptimized" in il.RoughlyOptimized.__module__

        b = self.tmp(self.lists, sample_num=3)
        is_success, p, minimum = b._compute_probabilities(
            self.lists,
            self.rankings,
        )

        b = il.RoughlyOptimized(self.lists, sample_num=3)
        c_is_success, c_p, c_minimum = b._compute_probabilities(
            self.lists,
            self.rankings,
        )

        self.assert_almost_equal(minimum, c_minimum)
        for i in range(len(p)):
            self.assert_almost_equal(p[i], c_p[i])

    def test_create_lp_matrices(self):
        l_l = len(self.lists)  # Number of original lists (rankers, teams)
        eta = len(self.rankings)  # Number of samples
        m_l = len(self.lists[0])  # Consider unbiasedness on top-(1..m_l)
        b = il.RoughlyOptimized(self.lists, sample_num=3)
        obj_c, const_A, const_b, const_G, const_h = b._create_lp_matrices(
            1.0, self.lists, self.rankings, eta, m_l, l_l)
        print(obj_c)
        assert obj_c.shape == (eta+m_l,)
        assert all(obj_c[eta:] == 1)
        print(const_A)
        assert const_A.shape == (1, eta+m_l)
        assert all(const_A[0][:eta] == 1)
        assert all(const_A[0][eta:] == 0)
        print(const_b)
        assert const_b.shape == (1,)
        assert all(const_b == 1)
        print(const_G)
        print(const_h)
        assert np.sum(const_G[0]) == -1
        assert np.sum(const_G[1]) == -1
        assert np.sum(const_G[2]) == -1
        assert all(const_h[:3] == 0)

        assert np.sum(const_G[3]) == 1
        assert np.sum(const_G[4]) == 1
        assert np.sum(const_G[5]) == 1
        assert all(const_h[3:6] == 1)

        assert all(const_G[6,:3] == -const_G[7,:3])
        assert all(const_G[6,3:] == const_G[7,3:])
        assert all(const_G[8,:3] == -const_G[9,:3])
        assert all(const_G[8,3:] == const_G[9,3:])

        d00, d01 = self.rankings[0]
        d10, d11 = self.rankings[1]
        d20, d21 = self.rankings[2]
        self.assert_almost_equal(const_G[6,0],
            self.rankings[0].credits[0][d00] - self.rankings[0].credits[1][d00])
        self.assert_almost_equal(const_G[6,1],
            self.rankings[1].credits[0][d10] - self.rankings[1].credits[1][d10])
        self.assert_almost_equal(const_G[6,2],
            self.rankings[2].credits[0][d20] - self.rankings[2].credits[1][d20])

        self.assert_almost_equal(const_G[8,0],
            self.rankings[0].credits[0][d00] - self.rankings[0].credits[1][d00]\
            + self.rankings[0].credits[0][d01] - self.rankings[0].credits[1][d01])
        self.assert_almost_equal(const_G[8,1],
            self.rankings[1].credits[0][d10] - self.rankings[1].credits[1][d10]\
            + self.rankings[1].credits[0][d11] - self.rankings[1].credits[1][d11])
        self.assert_almost_equal(const_G[8,2],
            self.rankings[2].credits[0][d20] - self.rankings[2].credits[1][d20]\
            + self.rankings[2].credits[0][d21] - self.rankings[2].credits[1][d21])

    def test_solution(self):
        b = il.RoughlyOptimized(self.lists, sample_num=3)
        is_success, p, minimum = b._compute_probabilities_loosely(self.lists, self.rankings)

        self.assert_almost_equal(p[0], 0.0)
        self.assert_almost_equal(p[1], 0.0)
        self.assert_almost_equal(p[2], 1.0)
        self.assert_almost_equal(b._lambdas[0], 0.5 - 1.0/3)
        self.assert_almost_equal(b._lambdas[1], 0.5 - 1.0/3 + 1.0/3 - 0.0)
        self.assert_almost_equal(
            minimum,
            np.sum(b._lambdas) + np.inner(p, b._sigmas),
        )

        b = self.tmp(self.lists, sample_num=3)
        is_success, p, original_minimum = b._compute_probabilities(
            self.lists,
            self.rankings,
        )
        self.assert_almost_equal(original_minimum, minimum)
