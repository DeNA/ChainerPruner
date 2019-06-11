import numpy as np
from chainerpruner.utils import calc_computational_cost

from tests.testutils import SimpleNet

def test_calc_computational_cost():
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    model = SimpleNet()
    cch = calc_computational_cost(model, x)
    assert True
