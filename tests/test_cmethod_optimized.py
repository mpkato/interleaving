import interleaving as il
from interleaving.cOptimized import Optimized
from .test_optimized import TestOptimized
import pytest

class TestCMethodOptimized(TestOptimized):
    @pytest.yield_fixture(autouse=True)
    def replace_method(self):
        tmp = il.Optimized
        il.Optimized = Optimized
        yield
        il.Optimized = tmp
