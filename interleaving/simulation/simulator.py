from .document import Document
from collections import defaultdict
from .ndcg import ndcg
import numpy as np

class Simulator(object):
    '''
    A simulator based on a dataset
    '''

    def __init__(self, dataset_filepath, query_sample_num, topk=10):
        '''
        dataset_filepath:
            path to the file including relevance grade and
            features in the format shown below:
            <line>    .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
            <target>  .=. <positive integer>
            <qid>     .=. <positive integer>
            <feature> .=. <positive integer>
            <value>   .=. <float>
            <info>    .=. <string>
        query_sample_num: the number of query samplings
        topk:             the number of documents shown to users in interleaving
        '''
        self.docs = defaultdict(list)
        with open(dataset_filepath) as f:
            for line in f:
                d = Document.readline(line)
                self.docs[d.qid].append(d)
        self.query_sample_num = query_sample_num
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

    def evaluate(self, rankers, user, method, sample_num=None):
        '''
        rankers: instances of Ranker to be compared
        user: an instance of User that is assumed in the simulation
        method: a class of intereaving method used in the simulation

        Return a dict indicating the number of pairs (i, j) 
        where i won j.
        '''
        methods = {}
        for q in self.docs:
            documents = self.docs[q]
            topk = self.topk if self.topk <= len(documents) else len(documents)
            ranked_lists = []
            for ranker in rankers:
                res = ranker.rank(documents)
                res = [id(d) for d in res]
                ranked_lists.append(res)
            methods[q] = method(ranked_lists,
                max_length=topk, sample_num=sample_num)

        result = []
        queries = np.random.choice(
            self.docs.keys(), self.query_sample_num, replace=True)
        for q in queries:
            documents = self.docs[q]
            rels = {id(d): d.rel for d in documents}
            ranking = methods[q].interleave()
            clicks = user.examine(ranking, rels)
            res = method.evaluate(ranking, clicks)
            result.append(res)
        return result

    def measure_error(self, il_result, ndcg_result):
        '''
        Return E_bin score in Schuth's CIKM 2014 paper.
        E_bin = sum_{i != j} sign(P^_{i, j} - 0.5) != sign(P_{i, j} - 0.5)
              / (# of rankers) * ((# of rankers) - 1),
              where P^_{i, j} = 1 (i won j) or 0 (j won i) in the interleaving,
              and P_{i, j} = 1 (i won j) or 0 (j won i) in terms fo nDCG.
        '''
        prefs = defaultdict(int)
        for res in il_result:
            for r in res:
                prefs[r] += 1

        result = 0.0
        for i in ndcg_result:
            for j in ndcg_result:
                paired_ndcg = ndcg_result[i] > ndcg_result[j]
                paired_pref = prefs[(i, j)] > prefs[(j, i)]
                if (paired_ndcg and not paired_pref)\
                    or (not paired_ndcg and paired_pref):
                    result += 1
        ranker_num = len(ndcg_result)
        result /= ranker_num * (ranker_num - 1)
        return result
