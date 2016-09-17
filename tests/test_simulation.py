import interleaving as il
import numpy as np
np.random.seed(0)

class TestSimulation(object):

    def test_simulator(self):
        sim = il.simulation.Simulator()
        ranker_a = il.simulation.NoisyRelevanceRanker(0.00001)
        ranker_b = il.simulation.NoisyRelevanceRanker(0.00001)
        user = il.simulation.User()
        res = sim.evaluate(ranker_a, ranker_b, user, il.TeamDraft())
        a_win, b_win, tie = res
        assert a_win == 0 and b_win == 0

    def test_ranker(self):
        sim = il.simulation.Simulator()
        ranker = il.simulation.NoisyRelevanceRanker(0.00001)
        ranks = ranker.rank(sim.documents, sim.relevance[0])
        assert set(sim.relevance[0]) == set(ranks[:10])
