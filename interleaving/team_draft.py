from .ranking import TeamRanking
from .interleaving_method import InterleavingMethod
import numpy as np

class TeamDraft(InterleavingMethod):
    '''
    Team Draft Interleaving
    '''
    def _sample(self, max_length, lists):
        '''
        Sample a ranking

        max_length: the maximum length of resultant interleaving
        *lists: lists of document IDs

        Return an instance of TeamDraftRanking
        '''
        result = TeamRanking(range(len(lists)))
        empty_teams = set()

        while len(result) < max_length:
            selected_team = self._select_team(result.teams, empty_teams)
            if selected_team is None:
                break
            docs = [x for x in lists[selected_team] if not x in result]
            if len(docs) > 0:
                selected_doc = docs[0]
                result.append(selected_doc)
                result.teams[selected_team].add(selected_doc)
            else:
                empty_teams.add(selected_team)

        return result

    def _select_team(self, teams, empty_teams):
        '''
        teams: a dict of team index and members (document IDs that belong to
               the team)
        empty_teams: a set of team indices that do not include available
                     documents

        Return a selected team index
        '''
        team_lens = [len(teams[i]) for i in teams if not i in empty_teams]
        if len(team_lens) == 0:
            return None
        min_team_num = min(team_lens)
        available_teams = [i for i in teams
            if len(teams[i]) == min_team_num and not i in empty_teams]
        if len(available_teams) == 0:
            return None
        selected_team = np.random.choice(available_teams)
        return selected_team

    @classmethod
    def _compute_scores(cls, ranking, clicks):
        '''
        ranking: an instance of Ranking
        clicks: a list of indices clicked by a user

        Return a list of scores of each ranker.
        '''
        return {i: len([c for c in clicks if ranking[c] in ranking.teams[i]])
            for i in ranking.teams}
