from .document import Document
from collections import defaultdict
from .ndcg import ndcg
import numpy as np

class Simulator(object):
    '''
    A simulator based on a dataset
    '''

    def __init__(self, dataset_filepath, num_per_query, topk=10):
        '''
        dataset_filepath: path to the file including relevance grade and
                          features in the format shown below:
            <line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
            <target> .=. <positive integer>
            <qid> .=. <positive integer>
            <feature> .=. <positive integer>
            <value> .=. <float>
            <info> .=. <string>
        num_per_query:    number of iteration of each query
        topk:             the number of documents shown to users in interleaving
        '''
        self.docs = defaultdict(list)
        with open(dataset_filepath) as f:
            for line in f:
                d = Document.readline(line)
                self.docs[d.qid].append(d)
        self.queries = list(self.docs.keys()) * num_per_query
        self.topk = topk

    def ndcg(self, rankers, cutoff):
        '''
        rankers: instances of Ranker
        cutoff:  cutoff for nDCG
        '''
        result = defaultdict(list)
        for q in self.docs:
            documents = self.docs[q]
            rels = {id(d): d.rel for d in documents}
            for idx, ranker in enumerate(rankers):
                res = ranker.rank(documents)
                ranked_list = [id(d) for d in res]
                score = ndcg(ranked_list, rels, cutoff)
                result[idx].append(score)
        for idx in result:
            result[idx] = np.average(result[idx])
        return result

    def evaluate(self, rankers, user, method):
        '''
        rankers: instances of Ranker to be compared
        user: an instance of User that is assumed in the simulation
        method: a class of intereaving method used in the simulation

        Return a dict indicating the number of pairs (i, j) 
        where i won j.
        '''
        result = defaultdict(int)
        for q in self.queries:
            documents = self.docs[q]
            rels = {id(d): d.rel for d in documents}
            ranked_lists = []
            for ranker in rankers:
                res = ranker.rank(documents)
                res = [id(d) for d in res]
                ranked_lists.append(res)
            ranking = method(ranked_lists, max_length=self.topk).interleave()
            clicks = user.examine(ranking, rels)
            res = method.evaluate(ranking, clicks)
            for r in res:
                result[r] += 1
        return result

    def measure_error(self, il_result, ndcg_result):
        result = 0.0
        for i in ndcg_result:
            for j in ndcg_result:
                paired_ndcg = ndcg_result[i] > ndcg_result[j]
                paired_pref = il_result[(i, j)] > il_result[(j, i)]
                if (paired_ndcg and not paired_pref)\
                    or (not paired_ndcg and paired_pref):
                    result += 1
        ranker_num = len(ndcg_result)
        result /= ranker_num * (ranker_num - 1)
        return result
