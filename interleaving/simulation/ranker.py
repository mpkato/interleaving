class Ranker(object):

    def __init__(self, scorer):
        self.scorer = scorer

    def rank(self, documents):
        result = sorted(documents,
            key=lambda x: self.scorer(x.features),
            reverse=True)
        return result
