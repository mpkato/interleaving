#!python
# cython: profile=True, boundscheck=False, wraparound=False

from .cOptimized import Optimized
import numpy as np
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False
import sys


class RoughlyOptimized(Optimized):
    '''
    Roughly Optimized Interleaving [Manabe et al. 2017]

    Args:
        lists: lists of document IDs
        max_length: the maximum length of resultant interleaving.
                    If this is None (default), it is set to the minimum length
                    of the given lists.
        sample_num: If this is None (default), an interleaved ranking is
                    generated every time when `interleave` is called.
                    Otherwise, `sample_num` rankings are sampled in the
                    initialization, one of which is returned when `interleave`
                    is called.
        credit_func: either 'inverse' (1/rank) or 'negative' (-rank)
        is_always_loose: always use the loose solution if True;
                    otherwise, try to solve the strict problem first,
                    and use the loose solution only if the strict one fails.
    '''

    def __init__(self, lists, max_length=None, sample_num=None,
        credit_func='inverse', is_always_loose=False):
        self._is_always_loose = is_always_loose
        super(RoughlyOptimized, self).__init__(lists,
            max_length=max_length, sample_num=sample_num,
            credit_func=credit_func)

    def _compute_probabilities(self, lists, rankings):
        if self._is_always_loose:
            return self._compute_probabilities_loosely(lists, rankings)

        try:
            is_success, x, f = super()._compute_probabilities(lists, rankings)
            if is_success:
                return is_success, x, f
        except ValueError:
            pass
        return self._compute_probabilities_loosely(lists, rankings)

    def _compute_probabilities_loosely(self,
                                       lists,
                                       rankings,
                                       bias_weight=1.0):
        l_l = len(lists)  # Number of original lists (rankers, teams)
        eta = len(rankings)  # Number of samples
        m_l = self.max_length  # Consider unbiasedness on top-(1..m_l)

        res = self._create_lp_matrices(
            bias_weight, lists, rankings, eta, m_l, l_l)
        obj_c, const_A, const_b, const_G, const_h = map(matrix, res)

        solution = solvers.lp(obj_c, const_G, const_h, const_A, const_b)

        # For test and bugfix
        self._sigmas = np.array(obj_c[:eta])
        self._sigmas = np.reshape(self._sigmas, (eta,))
        self._lambdas = np.array(solution['x'][eta:])
        self._lambdas = np.reshape(self._lambdas, (m_l,))

        p = np.array(solution['x'][:eta])
        p = np.reshape(p, (eta,))
        p = np.abs(p)
        p /= np.sum(p)

        return (
            solution['status'] == 'optimal',
            p,
            solution['primal objective']
        )

    def _create_lp_matrices(self, bias_weight, lists, rankings, eta, m_l, l_l):
        # Objective function
        ss = self._sensitivity(lists, rankings)
        alphas = bias_weight * np.ones(m_l)
        obj_c = np.hstack([ss, alphas])

        # Prob sum constraints
        const_A = np.hstack([np.ones((1, eta)), np.zeros((1, m_l))])
        const_b = np.ones(1)

        # Prob boundary constraints
        const_G0 = np.hstack([np.diag(-np.ones(eta)), np.zeros((eta, m_l))])
        const_h0 = np.zeros(eta)
        const_G1 = np.hstack([np.diag(np.ones(eta)), np.zeros((eta, m_l))])
        const_h1 = np.ones(eta)

        # Credit difference constraints
        credits = self._compute_credits_per_length(rankings, eta, m_l, l_l)
        const_Gd = self._compute_differences(credits)
        const_hd = np.zeros(const_Gd.shape[0])

        # Constraints
        const_G = np.vstack([const_G0, const_G1, const_Gd])
        const_h = np.hstack([const_h0, const_h1, const_hd])

        # Check the dimensions
        assert obj_c.shape == (eta + m_l,)
        assert const_A.shape == (1, eta + m_l)
        assert const_b.shape == (1,)
        assert const_G.shape == (2*eta+m_l*l_l*(l_l-1), eta + m_l)
        assert const_h.shape == (2*eta+m_l*l_l*(l_l-1),)

        return obj_c, const_A, const_b, const_G, const_h


    def _compute_credits_per_length(self, rankings, eta, m_l, l_l):
        credits = np.zeros((m_l, eta, l_l))
        for rid, ranking in enumerate(rankings):
            for idx, docid in enumerate(ranking):
                for team in ranking.credits:
                    credits[idx][rid][team] = ranking.credits[team][docid]
                    if idx > 0:
                        credits[idx][rid][team] += credits[idx - 1][rid][team]
        return credits

    def _compute_differences(self, credits):
        comb_num = credits.shape[2] * (credits.shape[2] - 1)
        result_row_num = credits.shape[0] * comb_num
        result = np.zeros((result_row_num, credits.shape[1]+credits.shape[0]))
        for idx in range(credits.shape[0]):
            base_rowidx = idx * comb_num
            inc_rowidx = 0
            for team_a in range(credits.shape[2]):
                for team_b in range(team_a+1, credits.shape[2]):
                    for rid in range(credits.shape[1]):
                        bias = credits[idx][rid][team_a] - credits[idx][rid][team_b]
                        # bias
                        result[base_rowidx+inc_rowidx][rid] = bias
                        result[base_rowidx+inc_rowidx+1][rid] = -bias
                    # lambda
                    result[base_rowidx+inc_rowidx][credits.shape[1]+idx] = -1
                    result[base_rowidx+inc_rowidx+1][credits.shape[1]+idx] = -1
                    inc_rowidx += 2
        return result
