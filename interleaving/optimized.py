from .ranking import Ranking
from .interleaving_method import InterleavingMethod
import numpy as np

class Optimized(InterleavingMethod):
    '''
    Optimized Interleaving
    '''
    def __init__(self, lists, max_length=None, sample_num=None):
        '''
        lists: lists of document IDs
        max_length: the maximum length of resultant interleaving.
                    If this is None (default), it is set to the minimum length
                    of the given lists.
        sample_num: If this is None (default), an interleaved ranking is
                    generated every time when `interleave` is called.
                    Otherwise, `sample_num` rankings are sampled in the
                    initialization, one of which is returned when `interleave`
                    is called.
        '''
        if sample_num is None:
            raise ValueError('sample_num cannot be None, '
                + 'i.e. the initial sampling is necessary')
        super(Optimized, self).__init__(lists,
            max_length=max_length, sample_num=sample_num)
        # self._rankings (sampled rankings) is obtained here
        self._probabilities = self._compute_probabilities(self._rankings)

    def _sample(self, max_length, lists):
        '''
        Prefix constraint sampling
        (Multileaved Comparisons for Fast Online Evaluation, CIKM'14)

        max_length: the maximum length of resultant interleaving
        *lists: lists of document IDs

        Return an instance of Ranking
        '''
        result = Ranking()
        teams = {}
        for i in range(len(lists)):
            teams[i] = set()
        empty_teams = set()

        while len(result) < max_length:
            selected_team = self._select_team(teams, empty_teams)
            if selected_team is None:
                break
            docs = [x for x in lists[selected_team] if not x in result]
            if len(docs) > 0:
                selected_doc = docs[0]
                result.append(selected_doc)
                teams[selected_team].add(selected_doc)
            else:
                empty_teams.add(selected_team)

        result.teams = teams
        return result

    def _select_team(self, teams, empty_teams):
        '''
        teams: a dict of team index and members (document IDs that belong to
               the team)
        empty_teams: a set of team indices that do not include available
                     documents

        Return a selected team index
        '''
        available_teams = [i for i in teams if not i in empty_teams]
        if len(available_teams) == 0:
            return None
        selected_team = np.random.choice(available_teams)
        return selected_team

    def _compute_probabilities(self, rankings):
        '''
        rankings: a list of Ranking instances

        Return a list of probabilities for input rankings
        '''
        raise NotImplementedError()

    @classmethod
    def _compute_scores(cls, ranking, clicks):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        return {i: len([c for c in clicks if ranking[c] in ranking.teams[i]])
            for i in ranking.teams}

