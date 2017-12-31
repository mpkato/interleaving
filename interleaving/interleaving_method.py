from collections import defaultdict
import json
import numpy as np


class InterleavingMethod(object):
    '''
    Abstract class for interleaving methods

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
    '''

    def __init__(self, lists, max_length=None, sample_num=None):
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
        '''
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = min([len(l) for l in lists])
        self.sample_num = sample_num
        self.lists = lists
        if self.sample_num:
            self._sample_rankings()

    def _sample_rankings(self):
        '''
        Sample `sample_num` rankings
        '''
        distribution = defaultdict(int)
        for i in range(self.sample_num):
            ranking = self._sample(self.max_length, self.lists)
            distribution[ranking] += 1.0 / self.sample_num
        self._rankings, self._probabilities = zip(*distribution.items())

    def _sample(self, max_length, lists):
        '''
        Sample a ranking

        max_length: the maximum length of resultant interleaving
        *lists: lists of document IDs

        Return an instance of Ranking
        '''
        raise NotImplementedError()

    def dump_rankings(self, file):
        '''
        Dump the sampled rankings into a file
        '''
        result = {}
        for rid, ranking in enumerate(self._rankings):
            result[hash(ranking)] = {
                'probability': self._probabilities[rid],
                'ranking': ranking.dumpd(),
            }
        with open(file, 'w') as f:
            json.dump(result, f, indent='    ')

    def interleave(self):
        '''
        Return an instance of Ranking
        '''
        if self.sample_num:
            self._probabilities = self._probabilities.astype(np.float64)
            self._probabilities /= np.sum(self._probabilities)
            i = np.argmax(np.random.multinomial(1, self._probabilities))
            return self._rankings[i]
        else:
            return self._sample(self.max_length, self.lists)

    @property
    def ranking_distribution(self):
        '''
        Return a list of Ranking and its probability
        if rankings are sampled in the initialization.
        Otherwise, return None.
        '''
        if self.sample_num:
            return zip(self._rankings, self._probabilities)
        else:
            return None

    @classmethod
    def evaluate(cls, ranking, clicks):
        '''
        Args:
            ranking: an instance of Ranking generated by Balanced.interleave
            clicks: a list of indices clicked by a user

        Returns:
            a list of pairs of ranker indices in which element (i, j)
            indicates i won j.
            e.g. a result [(1, 0), (2, 1), (2, 0)] indicates
            ranker 1 won ranker 0, and ranker 2 won ranker 0 as well as ranker 1.
        '''
        scores = cls.compute_scores(ranking, clicks)
        result = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if scores[i] > scores[j]:
                    result.append((i, j))
                elif scores[i] < scores[j]:
                    result.append((j, i))
                else:  # scores[i] == scores[j]
                    pass
        return result

    @classmethod
    def compute_scores(cls, ranking, clicks):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        raise NotImplementedError()
