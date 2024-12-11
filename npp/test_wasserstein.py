import numpy as np
from scipy import stats

from npp.wasserstein import Wasserstein


def test_wasserstein():

    np.random.seed(0)

    # Datasets drawn from same distribution. MMD should be approximately 0.
    for wass_p in [1, 2]:
        for j in range(3):
            x = np.random.randn(2000, 1)
            y = np.random.randn(2024, 1)
            wass = Wasserstein(wass_p)(x, y)
            assert np.allclose(wass, 0., atol=1e-1)

    # Datasets are internally the same, but completely different from one another.
    # Should achieve Wasserstein approximately equal to distance
    for wass_p in [1, 2]:
        x = np.zeros((2000, 1))
        y_dist = 1e8
        y = np.ones((1000, 1)) * y_dist
        wass = Wasserstein(wass_p)(x, y)
        assert np.allclose(wass, y_dist, rtol=1e-3)