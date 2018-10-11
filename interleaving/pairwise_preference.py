from .ranking import PairwisePreferenceRanking
from .interleaving_method import InterleavingMethod
import numpy as np

class PairwisePreference(InterleavingMethod):
    '''
    Pairwise Preference Multileaving
    '''
    def _sample(self, max_length, lists):
        '''
        Sample a ranking

        max_length: the maximum length of resultant interleaving
        lists: lists of document IDs

        Return an instance of PairwisePreferenceRanking
        '''
        result = PairwisePreferenceRanking(lists)

        while len(result) < max_length:
            focused_rank = len(result)
            candidates = self._current_candidates(lists, focused_rank, result)
            if not candidates:
                break

            # pairwise preference just performs uniform sampling from candidates
            sampled_index = np.random.randint(len(candidates))
            result.append(candidates[sampled_index])

        return result

    def _current_candidates(self, lists, focused_rank, selected_documents):
        """
        Candidates = { documents at a higher rank than `focused_rank` in any rankings  }
                     - { selected documents }
        """
        candidates = set()
        for l in lists:
            candidates.update(l[:focused_rank+1]) # documents at a higher rank

        candidates -= set(selected_documents) # exclude the selected documents
        return list(candidates)

    @classmethod
    def compute_scores(cls, ranking, clicks):
        scores = {i: 0.0 for i in range(len(ranking.lists))}
        preferences = cls._find_preferences(ranking, clicks)
        for sup, inf in preferences:
            r_under = cls._find_highest_rank_for_ranking(ranking, sup, inf)
            r_above = cls._find_highest_rank_for_all(ranking.lists, sup, inf)
            if r_under < r_above:
                # skip unless r_under >= r_above
                continue

            w = cls._compute_probability(r_above, ranking.lists, sup, inf)
            for i, r in enumerate(ranking.lists):
                sup_rank = cls._get_rank(r, sup)
                inf_rank = cls._get_rank(r, inf)
                if sup_rank < inf_rank: # sup >_r inf
                    scores[i] += 1 / w
                elif sup_rank > inf_rank: # sup <_r inf
                    scores[i] -= 1 / w
        return scores

    @classmethod
    def _find_preferences(cls, ranking, clicks):
        """
        Returns [(d_i, d_j)] where d_i is preferred to d_j
        """
        result = []
        if not clicks:
            # no preference if no click
            return result

        clicked_docs = {ranking[click] for click in clicks}
        for click in clicks:
            inferior_docs = []
            # a clicked document > documents ranked above it but not clicked
            above_unclicked = [doc for doc in ranking[:click]
                               if not doc in clicked_docs]
            inferior_docs += above_unclicked

            # a clicked document > a document ranked next to it but not clicked
            if click+1 < len(ranking):
                next_doc = ranking[click+1]
                if not next_doc in clicked_docs:
                    inferior_docs.append(next_doc)

            clicked_doc = ranking[click]
            result += [(clicked_doc, doc) for doc in inferior_docs]

        return result

    @classmethod
    def _find_highest_rank_for_all(cls, rankings, *docs):
        """
        max_{d} min_{r} r(d, r)
        """
        return cls._find_est_rank(max, min, rankings, *docs)

    @classmethod
    def _find_highest_rank_for_any(cls, rankings, *docs):
        """
        min_{d} min_{r} r(d, r)
        """
        return cls._find_est_rank(min, min, rankings, *docs)

    @classmethod
    def _find_highest_rank_for_ranking(cls, ranking, *docs):
        """
        min_{d} r(d, r)
        """
        return cls._find_est_rank(min, min, [ranking], *docs)

    @classmethod
    def _get_rank(cls, ranking, doc):
        try:
            rank = ranking.index(doc)
        except ValueError:
            rank = len(ranking)
        return rank

    @classmethod
    def _find_est_rank(cls, agg_for_docs, agg_for_rankings, rankings, *docs):
        """
        agg_for_docs_{d} agg_for_rankings_{r} r(d, r)
        """
        ranks = []
        for doc in docs:
            doc_ranks = []
            for ranking in rankings:
                rank = cls._get_rank(ranking, doc)
                doc_ranks.append(rank)
            doc_rank = agg_for_rankings(doc_ranks)
            ranks.append(doc_rank)
        result = agg_for_docs(ranks)
        return result

    @classmethod
    def _compute_probability(cls, r_above, rankings, *docs):
        w = 1.0
        min_x = cls._find_highest_rank_for_any(rankings, *docs)
        current_candidates = set()
        for r in rankings:
            current_candidates.update(r[:min_x])
        for x in range(min_x, r_above):
            for r in rankings:
                current_candidates.add(r[x])
            w *= 1 - 1 / (len(current_candidates) - x)
        return w
