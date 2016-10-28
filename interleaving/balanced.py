from .ranking import BalancedRanking
from .interleaving_method import InterleavingMethod
import numpy as np

class Balanced(InterleavingMethod):
    '''
    Balanced Interleaving
    '''
    def __init__(self, lists, max_length=None, sample_num=None):
        '''
        lists: two lists of document IDs
        max_length: the maximum length of resultant interleaving.
                    If this is None (default), it is set to the minimum length
                    of the given lists.
        sample_num: If this is None (default), an interleaved ranking is
                    generated every time when `interleave` is called.
                    Otherwise, `sample_num` rankings are sampled in the
                    initialization, one of which is returned when `interleave`
                    is called.
        '''
        if len(lists) != 2:
            raise ValueError('lists must be two rankings')
        super(Balanced, self).__init__(lists,
            max_length=max_length, sample_num=sample_num)

    def _sample(self, max_length, lists):
        '''
        Sample a ranking

        max_length: the maximum length of resultant interleaving
        *lists: lists of document IDs

        Return an instance of Ranking
        '''
        a, b = lists[0], lists[1]
        is_a_first = np.random.randint(0, 2) == 0
        result = BalancedRanking()
        k_a = 0
        k_b = 0
        while k_a < len(a) and k_b < len(b)\
            and len(result) < max_length:
            if (k_a < k_b) or (k_a == k_b and is_a_first):
                if not a[k_a] in result:
                    result.append(a[k_a])
                k_a += 1
            else:
                if not b[k_b] in result:
                    result.append(b[k_b])
                k_b += 1
        result.a = a
        result.b = b
        return result

    @classmethod
    def _compute_scores(cls, ranking, clicks):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        if len(clicks) == 0:
            return [0, 0]
        c_max = np.max(clicks)
        r_max = ranking[c_max]
        k_a = ranking.a.index(r_max) if r_max in ranking.a else len(ranking.a)
        k_b = ranking.b.index(r_max) if r_max in ranking.b else len(ranking.b)
        k = np.min([k_a, k_b])
        h_a = len([c for c in clicks if ranking[c] in ranking.a[:k+1]])
        h_b = len([c for c in clicks if ranking[c] in ranking.b[:k+1]])
        return [h_a, h_b]
