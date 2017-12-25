import interleaving as il
from interleaving.cProbabilistic import Probabilistic
from .test_probabilistic import TestProbabilistic
import pytest

class TestCMethodProbabilistic(TestProbabilistic):
    @pytest.yield_fixture(autouse=True)
    def replace_method(self):
        tmp = il.Probabilistic
        il.Probabilistic = Probabilistic
        yield
        il.Probabilistic = tmp
