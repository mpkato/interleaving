import interleaving as il
import numpy as np
import pytest
np.random.seed(0)
from .test_methods import TestMethods

class TestOptimized(TestMethods):

    def test_raise_value_error(self):
        with pytest.raises(ValueError):
            # None for `sample_num` is not allowed
            il.Optimized([[0], [1]])

    def test_init_sampling(self):
        b = il.Optimized([[1, 2], [2, 3]], sample_num=100)
