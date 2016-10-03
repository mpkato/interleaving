from .ranking import Ranking
from .interleaving_method import InterleavingMethod
import numpy as np
import copy

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
            for idx in self.doc_index[docid]:
                self.denominator -= self.numerators[idx]
                self._non_zero_index.remove(idx)
            if not self.denominator > 0:
                self.denominator = 0

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

    def __init__(self, tau, max_length, sample_num, *lists):
        '''
        tau: a parameter that determines the probability of documents
        max_length: the maximum length of resultant interleaving
        sample_num: If this is None, an interleaved ranking is generated
                    every time when `interleave` is called.
                    Otherwise, `sample_num` rankings are sampled in the
                    initialization, one of which is returned when `interleave`
                    is called.
        *lists: two lists of document IDs (no multileaving)
        '''
        super(Probabilistic, self).__init__(max_length, sample_num, *lists)
        self._softmaxs = {}
        for i, l in enumerate(lists):
            self._softmaxs[i] = self.Softmax(tau, l)

    def _sample(self, max_length, lists):
        '''
        Sample a ranking

        max_length: the maximum length of resultant interleaving
        *lists: lists of document IDs

        Return an instance of Ranking
        '''
        result = Ranking()
        ranker_indices = list(range(len(lists)))
        result.teams = {i: set() for i in ranker_indices}

        while len(result) < max_length and len(ranker_indices) > 0:
            rand_int = np.random.randint(0, len(ranker_indices))
            ranker_idx = ranker_indices[rand_int]
            docid = self._softmaxs[ranker_idx].sample()
            if docid is None:
                ranker_indices.remove(ranker_idx)
            else:
                result.append(docid)
                result.teams[ranker_idx].add(docid)
                for ranker_idx in ranker_indices:
                    if docid in self._softmaxs[ranker_idx].doc_index:
                        self._softmaxs[ranker_idx].delete(docid)

        # reset the state of softmax
        for i in self._softmaxs:
            self._softmaxs[i].reset()

        return result

    @classmethod
    def _compute_scores(cls, ranking, clicks):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        return {i: len([c for c in clicks if ranking[c] in ranking.teams[i]])
            for i in ranking.teams}

