class Ranker(object):
    '''
    A ranker that sorts documents by their features.
    '''
    def __init__(self, scorer):
        '''
        scorer: a function that takes a feature dict (Document.features)
            and returns a ranking score
        '''
        self.scorer = scorer

    def rank(self, documents):
        '''
        Documents are sorted by the descending order of `scorer(features)`.
        '''
        result = sorted(documents,
            key=lambda x: self.scorer(x.features),
            reverse=True)
        return result
