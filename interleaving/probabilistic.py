from .ranking import ProbabilisticRanking
from .interleaving_method import InterleavingMethod
import numpy as np

class Probabilistic(InterleavingMethod):
    '''
    Probabilistic Interleaving
    '''
    class Softmax(object):
        def __init__(self, tau, ranking):
            self.tau = tau
            self.ranking = ranking
            self.numerators = np.ndarray(len(ranking))
            self.doc_index = {}
            for r in range(len(ranking)):
                self.numerators[r] = 1.0 / (r+1) ** tau
                docid = ranking[r]
                if not docid in self.doc_index:
                    self.doc_index[docid] = set()
                self.doc_index[docid].add(r)
            self.denominator = np.sum(self.numerators)
            self._original_denominator = self.denominator
            self._non_zero_index = set(range(len(self.numerators)))

        def delete(self, docid):
            cum_numerators = 0.0
            old_denominator = self.denominator
            for idx in self.doc_index[docid]:
                cum_numerators += self.numerators[idx]
                self.denominator -= self.numerators[idx]
                self._non_zero_index.remove(idx)
            if not self.denominator > 0:
                self.denominator = 0
            # Returns probability of sampling docid before deletion
            if old_denominator <= 0:
                return 0.0
            else:
                return cum_numerators / old_denominator

        def reset(self):
            self.denominator = self._original_denominator
            self._non_zero_index = set(range(len(self.numerators)))

        def sample(self):
            if self.denominator == 0:
                return None
            p = np.random.rand() * self.denominator
            cum = 0.0
            for i in self._non_zero_index:
                cum += self.numerators[i]
                if cum > p:
                    return self.ranking[i]
            return self.ranking[i]

    def __init__(self, lists, max_length=None, sample_num=None,
        tau=3.0, replace=True):
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
        tau: a parameter that determines the probability of documents
             (default: 3.0)
        replace: rankings are sampled with replacement if it is True.
                          Otherwise, they are sampled without replacement,
                          e.g. given two rankings A and B, one of them is
                          sampled first and then another is used.
        '''
        self._softmaxs = {}
        self._replace = replace
        for i, l in enumerate(lists):
            self._softmaxs[i] = self.Softmax(tau, l)
        super(Probabilistic, self).__init__(lists,
            max_length=max_length, sample_num=sample_num)

    def _sample(self, max_length, lists):
        '''
        Sample a ranking

        max_length: the maximum length of resultant interleaving
        *lists: lists of document IDs

        Return an instance of Ranking
        '''
        ranker_indices = list(range(len(lists)))
        result = ProbabilisticRanking(lists)
        available_rankers = []

        while len(result) < max_length and len(ranker_indices) > 0:
            if len(available_rankers) == 0:
                available_rankers = list(ranker_indices)
                np.random.shuffle(available_rankers)
            if self._replace:
                ranker_idx = np.random.choice(available_rankers)
            else:
                ranker_idx = available_rankers.pop()
            docid = self._softmaxs[ranker_idx].sample()
            if docid is None:
                ranker_indices.remove(ranker_idx)
                available_rankers = list(ranker_indices)
            else:
                result.append(docid)
                for ranker_idx in ranker_indices:
                    if docid in self._softmaxs[ranker_idx].doc_index:
                        self._softmaxs[ranker_idx].delete(docid)

        # reset the state of softmax
        for i in self._softmaxs:
            self._softmaxs[i].reset()

        return result

    @classmethod
    def _compute_scores(cls, ranking, clicks, tau=3.0):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        if len(ranking.lists) == 2:
            result = {0: 0.0, 1: 0.0}
            num_assignments = 2 ** len(ranking)
            # ll = []
            for a in range(num_assignments):
                c = [0, 0]
                dists = [cls.Softmax(tau, l) for l in ranking.lists]
                cum_p = 1.0 / num_assignments
                # l = []
                for i, document_id in enumerate(ranking):
                    r = a % 2
                    # l.append(r)
                    if i in clicks:
                        c[r] += 1
                    cum_p *= dists[r].delete(document_id)
                    dists[(r + 1) % 2].delete(document_id)
                    a //= 2
                # ll.append([l, c[0], c[1], cum_p])
                if c[0] < c[1]:
                    result[1] += cum_p
                elif c[1] < c[0]:
                    result[0] += cum_p
            # for l in sorted(ll):
                # print(l)
            return result
        # if 2 < len(original_lists):
            # TODO: Evaluation of multileaved result
        else:
            raise ValueError('invalid number of original lists')

