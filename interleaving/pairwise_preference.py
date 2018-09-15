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
        # TODO: implementation!!!
        raise NotImplementedError()
