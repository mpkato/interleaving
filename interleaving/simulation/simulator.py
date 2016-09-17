import numpy as np
from collections import defaultdict

class Simulator(object):

    def __init__(self, query_num=100, doc_num=1000,
        rel_doc_dist=[0.01], topk=10):
        self.query_num = query_num
        self.topk = topk
        self.documents = range(doc_num)
        self.relevance = defaultdict(dict)
        for q in range(self.query_num):
            self.relevance[q] = self._init_relevance(doc_num, rel_doc_dist)

    def _init_relevance(self, doc_num, rel_doc_dist):
        result = {}
        sorted_documents = list(self.documents)
        np.random.shuffle(sorted_documents)
        start = 0
        for idx, dist in enumerate(rel_doc_dist):
            grade = idx + 1
            num = int(doc_num * dist)
            for r in sorted_documents[start:start+num]:
                result[r] = grade
        return result

    def evaluate(self, ranker_a, ranker_b, user, method):
        a_win, b_win, tie = 0, 0, 0
        for q in range(self.query_num):
            rels = self.relevance[q]
            a = ranker_a.rank(self.documents, rels)
            b = ranker_b.rank(self.documents, rels)
            ranking = method.interleave(a[:self.topk], b[:self.topk])
            clicks = user.examine(ranking, rels)
            res = method.evaluate(ranking, clicks)
            if res[0] > 0:
                a_win += 1
            elif res[1] > 0:
                b_win += 1
            else:
                tie += 1
        return a_win, b_win, tie
