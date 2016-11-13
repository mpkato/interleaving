import interleaving as il
import numpy as np
np.random.seed(0)

class TestSimulation(object):

    def test_simulator(self):
        sim = il.simulation.Simulator("./MQ2008/Fold1/train.txt", 1)

        bm25 = il.simulation.Ranker(lambda x: x[25])
        lm = il.simulation.Ranker(lambda x: x[40])
        pr = il.simulation.Ranker(lambda x: x[41])
        rankers = [bm25, lm, pr]

        user = il.simulation.User(click_probs=[0.0, 0.5, 1.0],
            stop_probs=[0.0, 0.0, 0.0])

        result = sim.evaluate(rankers, user, il.TeamDraft)
        assert result[(1, 0)] > result[(0, 1)]
        assert result[(1, 2)] > result[(2, 1)]
        assert result[(0, 2)] > result[(2, 0)]

    def test_simulator_ndcg(self):
        sim = il.simulation.Simulator("./MQ2008/Fold1/train.txt", 1)

        bm25 = il.simulation.Ranker(lambda x: x[25])
        lm = il.simulation.Ranker(lambda x: x[40])
        pr = il.simulation.Ranker(lambda x: x[41])
        rankers = [bm25, lm, pr]

        ndcgs = sim.ndcg(rankers, 10)

    def test_measure_error(self):
        sim = il.simulation.Simulator("./MQ2008/Fold1/train.txt", 10)

        tfidf = il.simulation.Ranker(lambda x: x[15])
        bm25 = il.simulation.Ranker(lambda x: x[25])
        lm = il.simulation.Ranker(lambda x: x[40])
        pr = il.simulation.Ranker(lambda x: x[41])
        rankers = [tfidf, bm25, lm, pr]

        user = il.simulation.User(click_probs=[0.4, 0.7, 0.9],
            stop_probs=[0.1, 0.3, 0.5])

        ndcg_result = sim.ndcg(rankers, 10)

        il_result = sim.evaluate(rankers, user, il.TeamDraft)
        error = sim.measure_error(il_result, ndcg_result)
