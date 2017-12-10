from .document import Document
from collections import defaultdict
from .ndcg import ndcg
import numpy as np

class Simulator(object):
    '''
    A simulator based on a learning to rank dataset.

    Args:
        dataset_filepaths:
            paths to the file including relevance grade and
            features in the format shown below:
            <line>    .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
            <target>  .=. <positive integer>
            <qid>     .=. <positive integer>
            <feature> .=. <positive integer>
            <value>   .=. <float>
            <info>    .=. <string>
        usertypes:        a dict of {usertype: simulation.User}
        query_sample_num: the number of query samplings
        topk:             the number of documents shown to users in interleaving
    '''

    def __init__(self, dataset_filepaths, usertypes, query_sample_num, topk):
        self.docs = defaultdict(list)
        for dataset_filepath in dataset_filepaths:
            with open(dataset_filepath) as f:
                for line in f:
                    d = Document.readline(line)
                    self.docs[d.qid].append(d)
        self.usertypes = usertypes
        self.query_sample_num = query_sample_num
        self.topk = topk

    def ndcg(self, rankers, cutoff):
        '''
        Args:
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

    def run(self, rankers, method):
        '''
        Args:
            rankers: instances of Ranker to be compared
            user: an instance of User that is assumed in the simulation
            method: a class of intereaving method used in the simulation

        Returns:
            Return a list of dicts storing the score of each ranker.
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
            methods[q] = method(ranked_lists, max_length=topk)

        result = defaultdict(list)
        queries = np.random.choice(list(self.docs.keys()),
            self.query_sample_num, replace=True)
        for q in queries:
            documents = self.docs[q]
            rels = {id(d): d.rel for d in documents}
            ranking = methods[q].interleave()
            for usertype, user in self.usertypes.items():
                clicks = user.examine(ranking, rels)
                res = method.compute_scores(ranking, clicks)
                result[usertype].append(res)
        return result

    @classmethod
    def measure_error(cls, il_result, ndcg_result):
        '''
        Return E_bin score in Schuth's CIKM 2014 paper.
        E_bin = sum_{i != j} sign(P^_{i, j} - 0.5) != sign(P_{i, j} - 0.5)
              / (# of rankers) * ((# of rankers) - 1),
              where P^_{i, j} = 1 (i won j) or 0 (j won i) in the interleaving,
              and P_{i, j} = 1 (i won j) or 0 (j won i) in terms fo nDCG.
        '''

        prefs = defaultdict(int)
        for scores in il_result:
            for i in range(len(scores)):
                for j in range(i+1, len(scores)):
                    if scores[i] > scores[j]:
                        prefs[(i, j)] += 1
                    elif scores[i] < scores[j]:
                        prefs[(j, i)] += 1
                    else: # scores[i] == scores[j]
                        pass

        result = 0.0
        for i in ndcg_result:
            for j in ndcg_result:
                if i == j:
                    continue
                cond = (ndcg_result[i] - ndcg_result[j])\
                    * (prefs[(i, j)] - prefs[(j, i)])
                if cond < 0: # a different pairwise preference
                    result += 1
                elif cond == 0:
                    # for preciseness
                    if not ((ndcg_result[i] == ndcg_result[j])\
                        and (prefs[(i, j)] == prefs[(j, i)])):
                        result += 1
        ranker_num = len(ndcg_result)
        result /= ranker_num * (ranker_num - 1)
        return result
