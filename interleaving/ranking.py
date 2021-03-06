from collections import defaultdict

class BalancedRanking(list):
    '''
    A list of document IDs generated by an interleaving method
    including two rankers A and B
    '''
    __slots__ = ['a', 'b']
    def __hash__(self):
        return hash((tuple(self), tuple(self.a), tuple(self.b)))

    def dumpd(self):
        return {
            'a': self.a,
            'b': self.b,
            'ranking_list': self,
        }


class CreditRanking(list):
    '''
    A list of document IDs generated by an interleaving method
    including credits

    Args:
        num_rankers: number of rankers
        contents:    initial list of document IDs (optional)
    '''
    __slots__ = ['credits']
    def __init__(self, num_rankers, contents=[]):
        '''
        Initialize self.credits

        num_rankers: number of rankers
        contents:    initial list of document IDs (optional)
        '''
        self += contents
        self.credits = {}
        for i in range(num_rankers):
            self.credits[i] = defaultdict(float)

    def __hash__(self):
        l = []
        for k, v in self.credits.items():
            ll = []
            for kk, vv in v.items():
                ll.append((kk, vv))
            l.append((k, frozenset(ll)))
        return hash((tuple(self), frozenset(l)))

    def dumpd(self):
        return {
            'credits': self.credits,
            'ranking_list': self,
        }


class ListsRanking(list):
    '''
    A list of document IDs generated by an interleaving method,
    including original rankings

    Args:
        lists:    list of original document ID lists
        contents: initial list of document IDs (optional)
    '''
    __slots__ = ['lists']
    def __init__(self, lists, contents=[]):
        '''
        Initialize self.teams

        lists:    list of original document ID lists
        contents: initial list of document IDs (optional)
        '''
        self += contents
        self.lists = lists

    def __hash__(self):
        l = []
        for v in self.lists:
            l.append(tuple(v))
        return hash((tuple(self), tuple(l)))

    def dumpd(self):
        return {
            'ranking_list': self,
            'lists': self.lists,
        }

class ProbabilisticRanking(ListsRanking):
    pass

class TeamRanking(list):
    '''
    A list of document IDs generated by an interleaving method,
    including teams

    Args:
        team_indices: indices for self.teams
        contents:     initial list of document IDs (optional)
    '''
    __slots__ = ['teams']
    def __init__(self, team_indices, contents=[]):
        '''
        Initialize self.teams

        team_indices: indices for self.teams
        contents:     initial list of document IDs (optional)
        '''
        self += contents
        self.teams = {i: set() for i in team_indices}

    def __hash__(self):
        '''
        TeamRanking can be a key by which
        rankings with the same document ID list
        and the same team assignment are the same
        '''
        l = []
        for k, v in self.teams.items():
            l.append((k, frozenset(v)))
        return hash((tuple(self), frozenset(l)))

    def dumpd(self):
        team_dict = {}
        for tid, s in self.teams.items():
            team_dict[tid] = sorted(list(s))
        return {
            'ranking_list': self,
            'teams': team_dict,
        }

class PairwisePreferenceRanking(ListsRanking):
    pass
