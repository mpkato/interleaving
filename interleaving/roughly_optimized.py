from .optimized import Optimized
import numpy as np
import pulp
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
        bias_weight: the weight for the bias in the objective function
        is_always_loose: always use the loose solution if True;
                    otherwise, try to solve the strict problem first,
                    and use the loose solution only if the strict one fails.
    '''
    def __init__(self, lists, max_length=None, sample_num=None,
        credit_func='inverse', bias_weight=1.0, is_always_loose=False):
        self.bias_weight = bias_weight
        self._is_always_loose = is_always_loose
        super(RoughlyOptimized, self).__init__(lists,
            max_length=max_length, sample_num=sample_num,
            credit_func=credit_func)

    def _compute_probabilities(self, lists, rankings):
        if not self._is_always_loose:
            try:
                is_success, x, f = super()._compute_probabilities(lists, rankings)
                if is_success:
                    return is_success, x, f
            except ValueError:
                pass
        return self._compute_probabilities_loosely(lists, rankings, self.bias_weight)

    def _compute_probabilities_loosely(self,
                                       lists,
                                       rankings,
                                       bias_weight):
        l_l = len(lists)  # Number of original lists (rankers, teams)
        eta = len(rankings)  # Number of samples
        m_l = self.max_length  # Consider unbiasedness on top-(1..m_l)
        ps, ls = [], []  # Probabilities and lambdas

        # Problem
        prob = pulp.LpProblem("_cpl", pulp.LpMinimize)

        # Variables and boundaries
        for i in range(eta):
            ps.append(pulp.LpVariable("p(ranking_%i)" % i,
                                      lowBound=0.0,
                                      upBound=1.0))
        for k in range(m_l):
            ls.append(pulp.LpVariable("lambda_%i" % k,
                                      lowBound=0.0))

        # Objective function
        terms = []
        ss = self._sensitivity(lists, rankings)
        for (s, p) in zip(ss, ps):
            terms.append(s * p)
        for l in ls:
            terms.append(bias_weight * l)
        prob += pulp.lpSum(terms)

        # Inequarity constraints
        credits = []
        for _ in range(m_l):
            credits.append([[0 for _ in range(l_l)] for _ in range(eta)])
        for rid, ranking in enumerate(rankings):
            for idx, docid in enumerate(ranking):
                for team in ranking.credits:
                    credits[idx][rid][team] = ranking.credits[team][docid]
                    if idx > 0:
                        credits[idx][rid][team] += credits[idx - 1][rid][team]
        for idx in range(m_l):
            base_terms = [-1.0 * ls[idx]]
            for team_a in range(l_l):
                for team_b in range(l_l):
                    if team_a == team_b:
                        continue
                    terms = base_terms[:]
                    for rid in range(eta):
                        bias = credits[idx][rid][team_a] - credits[idx][rid][team_b]
                        terms.append(bias * ps[rid])
                    label = "%ivs.%i@%i" % (team_a, team_b, idx)
                    prob += pulp.lpSum(terms) <= 0.0, label

        # Equarity constraint
        prob += pulp.lpSum(ps) == 1.0

        prob.solve()

        # For test and bugfix
        self._problem = prob
        self._sigmas = ss
        self._lambdas = np.array([l.varValue for l in ls])

        return (
            pulp.LpStatus[prob.status] == 'Optimal',
            np.array([p.varValue for p in ps]),
            pulp.value(prob.objective)
        )
