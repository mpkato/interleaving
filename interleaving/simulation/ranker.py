import numpy as np

class Ranker(object):
    '''
    A ranker for simulation.
    '''
    def rank(self, documents, relevance):
        '''
        documents: a list of document IDs
        relevance: a dict of relevance for each document ID
        '''
        raise NotImplementedError()

class NoisyRelevanceRanker(Ranker):

    def __init__(self, noise):
        '''
        noise: the variance of a normal distribution used as noise to
        relevance scores.
        '''
        self.noise = noise

    def rank(self, documents, relevance):
        '''
        documents: a list of document IDs
        relevance: a dict of relevance for each document ID

        Return a ranked list of document IDs that are sorted 
        by given relevance with noise generated from a normal distribution.
        '''
        scores = np.array([-relevance.get(d, 0) for d in documents],
            dtype=np.float64)
        n = np.random.normal(0, self.noise, len(scores))
        scores += n
        result = [documents[i] for i in list(np.argsort(scores))]
        return result
