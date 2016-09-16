from .ranking import Ranking
import numpy as np

class TeamDraft(object):
    def interleave(self, a, b):
        result = Ranking()
        team_a = []
        team_b = []
        while len(set(a) - set(result)) > 0 and len(set(b) - set(result)) > 0:
            is_a_first = np.random.randint(0, 2) == 0
            if len(team_a) < len(team_b) or\
                (len(team_a) == len(team_b) and is_a_first):
                k = [i for i, x in enumerate(a) if not x in result][0]
                result.append(a[k])
                team_a.append(a[k])
            else:
                k = [i for i, x in enumerate(b) if not x in result][0]
                result.append(b[k])
                team_b.append(b[k])
        result.team_a = team_a
        result.team_b = team_b
        return result

    def evaluate(self, ranking, clicks):
        if len(clicks) == 0:
            return (0, 0)
        h_a = len([c for c in clicks if ranking[c] in ranking.team_a])
        h_b = len([c for c in clicks if ranking[c] in ranking.team_b])
        if h_a > h_b:
            return (1, 0)
        elif h_b > h_a:
            return (0, 1)
        else:
            return (0, 0)
