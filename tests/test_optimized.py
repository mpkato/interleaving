import interleaving as il
import numpy as np
import pytest
np.random.seed(0)
from .test_methods import TestMethods

class TestOptimized(TestMethods):

    def test_raise_value_error(self):
        with pytest.raises(ValueError):
            # None for `sample_num` is not allowed
            il.Optimized(1, None, [0], [1])

    def test_init_sampling(self):
        b = il.Optimized(2, 100, [1, 2], [2, 3])
