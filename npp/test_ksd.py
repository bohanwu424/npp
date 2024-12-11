import numpy as np
from scipy import stats

from npp.ksd import KSD


def test_ksd():

    np.random.seed(0)

    # Data drawn from same distribution as model. KSD should be approximately 0.
    model = stats.norm(loc=np.array([2., 4.]), scale=np.array([0.5, 1.]))
    for j in range(3):
        x = model.rvs((2000, 2))
        ksd = KSD()(model, x)
        assert np.allclose(ksd, 0., atol=3e-3)


    # Data very, very far from model. KSD should be very large
    model = stats.norm(loc=np.array([2., 4.]), scale=np.array([0.5, 1.]))
    for j in range(3):
        x = model.rvs((2000, 2)) + 1e24
        ksd = KSD()(model, x)
        assert ksd > 1e24