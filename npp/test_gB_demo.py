import numpy as np

from npp.gB_demo import weighted_quantile


def test_weighted_median():
    vals = np.array([1, 3, 2, 4])
    weights = np.zeros((4, 3))
    weights[:, 0] = np.array([0.25, 0., 0.5, 0.25])
    weights[:, 1] = np.array([0.25, 0.5, 0., 0.25])
    weights[:, 2] = np.array([0.5, 0., 0.5, 0.])
    tst_med = weighted_quantile(0.5, vals, weights)
    assert np.allclose(tst_med, np.array([2, 3, 1.5]))
