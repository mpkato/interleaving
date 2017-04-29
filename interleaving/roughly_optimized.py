from .optimized import Optimized
import numpy as np
import pulp
import sys


class RoughlyOptimized(Optimized):
    '''x = (p_1, p_2, ..., p_eta, lambda_1, lambda_2, ..., lambda_n_l)'''
    def _compute_probabilities(self, lists, rankings):
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
