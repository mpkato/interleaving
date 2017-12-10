import os
import pytest
import math
import interleaving as il
import numpy as np
from collections import defaultdict

class TestSimulation(object):

    def test_simulator_run(self, data_filepaths, user):
        sim = il.simulation.Simulator(data_filepaths, user, 10, 10)
        m1 = il.simulation.Ranker(lambda x: x[1])
        m2 = il.simulation.Ranker(lambda x: x[2])
        m3 = il.simulation.Ranker(lambda x: x[3])
        rankers = [m1, m2, m3]
        res = sim.run(rankers, il.TeamDraft)
        res = res["default"]
        result = defaultdict(int)
        assert len(res) == 10
        for r in res:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    if r[i] > r[j]:
                        result[(i, j)] += 1
                    elif r[j] > r[i]:
                        result[(j, i)] += 1
        assert result[(0, 1)] > result[(1, 0)]
        assert result[(0, 2)] > result[(2, 0)]
        assert result[(1, 2)] > result[(2, 1)]

    def test_simulator_ndcg(self, data_filepaths):
        sim = il.simulation.Simulator(data_filepaths, None, None, None)

        m1 = il.simulation.Ranker(lambda x: x[1])
        m2 = il.simulation.Ranker(lambda x: x[2])
        m3 = il.simulation.Ranker(lambda x: x[3])
        rankers = [m1, m2, m3]

        ndcgs = sim.ndcg(rankers, 2)
        assert ndcgs[0] == 1
        ndcg_1 = (1.0 + 2.0 / np.log2(3.0)) / (2.0 + 1.0 / np.log2(3.0))
        assert np.abs(ndcgs[1] - ndcg_1) < 10e-10
        ndcg_2 = (1.0 / np.log2(3.0)) / (2.0 + 1.0 / np.log2(3.0))
        assert np.abs(ndcgs[2] - ndcg_2) < 10e-10

    def test_measure_error(self, data_filepaths):
        # 2 > 1 > 0
        il_result = [{0: 0, 1: 1, 2: 2}]
        ndcg_result = {0: 0.2, 1: 0.1, 2: 0.3}

        error = il.simulation.Simulator.measure_error(il_result, ndcg_result)
        assert error == 2 / 6

    @pytest.fixture
    def data_filepaths(self):
        # 1 > 2 > 3
        return [os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "fixtures", "l2r_sample.txt")]

    @pytest.fixture
    def user(self):
        return {"default": il.simulation.User(click_probs=[0.0, 0.5, 1.0],
            stop_probs=[0.0, 0.0, 0.0])}
