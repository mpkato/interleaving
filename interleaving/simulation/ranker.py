class Ranker(object):
    '''
    A ranker that sorts documents by their features.
    Args:
        scorer: a function that takes a feature dict (Document.features)
            and returns a ranking score
    '''
    def __init__(self, scorer):
        self.scorer = scorer

    def rank(self, documents):
        '''
        Args:
            documents: a list of document IDs
            relevance: a dict of relevance for each document ID

        Returns:
            a ranked list of document IDs that are sorted
            by given relevance with noise generated from a normal distribution.
        '''
        result = sorted(documents,
            key=lambda x: self.scorer(x.features),
            reverse=True)
        return result
