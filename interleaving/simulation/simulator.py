import numpy as np
from collections import defaultdict

class Simulator(object):
    '''
    A simulator that generates psuedo queries, documents, and qrels,
    conducts interleaving, simulates user clicks, and evaluates rankers.
    '''

    def __init__(self, query_num=100, doc_num=1000,
        rel_doc_dist=[0.01], topk=10):
        '''
        query_num: the number of queries used in the simulation
                   (queries are used only once in a simulation)
        doc_num: the number of documents used in the simulation
        rel_doc_dist: a list of probability values for grades.
                      [p1, p2, ..., pn] indicates that the maximum grade is n,
                      and the probability of grade i is pi (that of grade
                      0 is 1 - sum(p1, p2, ..., pn)).
        topk: the number of documents shown to users in interleaving
        '''
        self.query_num = query_num
        self.topk = topk
        self.documents = range(doc_num)
        self.relevance = defaultdict(dict)
        for q in range(self.query_num):
            self.relevance[q] = self._init_relevance(doc_num, rel_doc_dist)

    def _init_relevance(self, doc_num, rel_doc_dist):
        '''
        Generates psuedo qrels based on rel_doc_dist, a list of probability
        values for grades.
        '''
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
        '''
        ranker_a: an instance of Ranker to be compared
        ranker_b: an instance of Ranker to be compared
        user: an instance of User that is assumed in the simulation
        method: a class of intereaving method used in the simulation

        Return a_win (how many times a won), b_win (how_many_times b won), 
        and tie (the number of ties in the interleaving).
        '''
        a_win, b_win, tie = 0, 0, 0
        for q in range(self.query_num):
            rels = self.relevance[q]
            a = ranker_a.rank(self.documents, rels)
            b = ranker_b.rank(self.documents, rels)
            ranking = method([a, b], max_length=self.topk).interleave()
            clicks = user.examine(ranking, rels)
            res = method.evaluate(ranking, clicks)
            if (0, 1) in res:
                a_win += 1
            elif (1, 0) in res:
                b_win += 1
            else:
                tie += 1
        return a_win, b_win, tie
