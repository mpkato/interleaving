import numpy as np

class NoisyRelevanceRanker(object):

    def __init__(self, noise):
        self.noise = noise

    def rank(self, documents, relevance):
        scores = np.array([-relevance.get(d, 0) for d in documents],
            dtype=np.float64)
        n = np.random.normal(0, self.noise, len(scores))
        scores += n
        result = list(np.argsort(scores))
        return result
