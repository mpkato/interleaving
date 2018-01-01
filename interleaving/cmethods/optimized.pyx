#!python
# cython: profile=True, boundscheck=False, wraparound=False

from .ranking import CreditRanking
from .interleaving_method import InterleavingMethod
import numpy as np
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


class Optimized(InterleavingMethod):
    '''
    Optimized Interleaving

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
    '''

    def __init__(self, lists, max_length=None, sample_num=None,
        credit_func='inverse'):
        '''
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
        '''
        if sample_num is None:
            raise ValueError('sample_num cannot be None, '
                + 'i.e. the initial sampling is necessary')
        if credit_func == 'inverse':
            self._credit_func = lambda x: 1.0 / x
        elif credit_func == 'negative':
            self._credit_func = lambda x: -x
        else:
            raise ValueError('credit_func should be either inverse or negative')
        super(Optimized, self).__init__(lists,
            max_length=max_length, sample_num=sample_num)
        # self._rankings (sampled rankings) is obtained here
        res = self._compute_probabilities(lists, self._rankings)
        is_success, self._probabilities, _ = res
        if not is_success:
            raise ValueError('Optimization failed')

    def _sample_rankings(self):
        '''
        Sample `sample_num` rankings
        '''
        distribution = {}
        for i in range(self.sample_num):
            ranking = self._sample(self.max_length, self.lists)
            existing_prob = distribution.get(ranking, 0.0)
            distribution[ranking] = existing_prob + 1.0 / self.sample_num
        self._rankings, self._probabilities = zip(*distribution.items())


    def _sample(self, max_length, lists):
        '''
        Prefix constraint sampling
        (Multileaved Comparisons for Fast Online Evaluation, CIKM'14)

        max_length: the maximum length of resultant interleaving
        lists: lists of document IDs

        Return an instance of Ranking
        '''
        num_rankers = len(lists)
        result = CreditRanking(num_rankers)
        teams = set(range(num_rankers))

        while len(result) < max_length:
            if len(teams) == 0:
                break
            selected_team = np.random.choice(list(teams))
            docs = [x for x in lists[selected_team] if not x in result]
            if len(docs) > 0:
                selected_doc = docs[0]
                result.append(selected_doc)
            else:
                teams.remove(selected_team)

        # assign credits
        for docid in result:
            for team in result.credits:
                if docid in lists[team]:
                    rank = lists[team].index(docid) + 1
                else:
                    rank = len(lists[team]) + 1
                result.credits[team][docid] = self._credit_func(rank)

        return result

    def _compute_probabilities(self, lists, rankings):
        '''
        Solve the optimization problem in
        (Multileaved Comparisons for Fast Online Evaluation, CIKM'14)

        lists: lists of document IDs
        rankings: a list of Ranking instances

        Return a list of probabilities for input rankings
        '''
        # probability constraints
        A_p_sum = np.array([1.0]*len(rankings))
        # unbiasedness constraints
        ub_cons = self._unbiasedness_constraints(lists, rankings)
        ub_cons = ub_cons[np.invert(np.all(ub_cons == 0, axis=1))]

        # sensitivity
        sensitivity = self._sensitivity(lists, rankings)
        sensitivity = np.reshape(sensitivity, (sensitivity.shape[0], 1))

        # constraints
        if ub_cons.shape[0] > 0:
            # ub_cons exist
            A_eq = np.vstack((A_p_sum, ub_cons))
        else:
            # ub_cons is empty
            A_eq = np.reshape(A_p_sum, (1, A_p_sum.shape[0]))
        b_eq = np.array([1.0] + [0.0]*ub_cons.shape[0])

        G_0 = np.diag(-np.ones(len(rankings)))
        h_0 = np.zeros(len(rankings))
        G_1 = np.diag(np.ones(len(rankings)))
        h_1 = np.ones(len(rankings))

        G = np.vstack((G_0, G_1))
        h = np.hstack((h_0, h_1))

        sensitivity, G, h, A_eq, b_eq = map(matrix, (sensitivity, G, h, A_eq, b_eq))

        # solving the optimization problem
        solution = solvers.lp(sensitivity, # objective function
            G, h, # constraints
            A_eq, b_eq # 0 <= p <= 1
            )

        p = np.array(solution['x'])
        p = np.reshape(p, (len(rankings),))
        p = np.abs(p)
        p /= np.sum(p)

        return (
            solution['status'] == 'optimal',
            p,
            solution['primal objective']
        )

    def _unbiasedness_constraints(self, lists, rankings):
        '''
        for each k and team x, for a certain c_k:
            sum_{L_i} {p_i} * sum^k_{j=1} ranking.credits[x][d_j] = c_k
        In other words,
            sum_{L_i} {p_i} * sum^k_{j=1}
                (ranking.credits[x][d_j] - ranking.credits[x+1][d_j]) = 0
        '''
        result = []
        credits = np.zeros((self.max_length, len(rankings), len(lists)))
        for rid, ranking in enumerate(rankings):
            for idx, docid in enumerate(ranking):
                for team in ranking.credits:
                    credits[idx, rid, team] = ranking.credits[team][docid]
                    if idx > 0:
                        credits[idx, rid, team] += credits[idx-1, rid, team]
        for i in range(len(lists) - 1):
            diff = credits[:, :, i] - credits[:, :, i+1]
            result.append(diff)
        result = np.vstack(result)
        return result

    def _sensitivity(self, lists, rankings):
        '''
        Expected variance
        '''
        # compute the mean of each ranking
        mu = np.zeros(len(rankings))
        for rid, ranking in enumerate(rankings):
            for idx, docid in enumerate(ranking):
                click_prob = 1.0 / (idx + 1)
                credit = np.sum(
                    [ranking.credits[x][docid] for x in ranking.credits])
                mu[rid] += click_prob * credit
        mu /= len(lists)

        # compute the variance
        var = np.zeros(len(rankings))
        for rid, ranking in enumerate(rankings):
            for x in ranking.credits:
                v = 0.0
                for idx, docid in enumerate(ranking):
                    click_prob = 1.0 / (idx + 1)
                    if docid in ranking.credits[x]:
                        v += click_prob * ranking.credits[x][docid]
                v -= mu[rid]
                var[rid] += v ** 2

        return var

    @classmethod
    def compute_scores(cls, ranking, clicks):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        return {i: sum([ranking.credits[i][ranking[c]] for c in clicks])
            for i in ranking.credits}
