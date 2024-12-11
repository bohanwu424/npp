import numpy as np

from npp.mmd import MMD


def test_mmd():

    np.random.seed(0)

    # Datasets drawn from same distribution. MMD should be approximately 0.
    for j in range(3):
        x = np.random.randn(1000, 3)
        y = np.random.randn(1200, 3)
        mmd = MMD()(x, y)
        assert np.allclose(mmd, 0., atol=1e-3, rtol=1e-3)

    # Datasets are internally the same, but completely different from one another.
    # Should achieve MMD approximately 2
    x = np.ones((100, 1))
    y = np.ones((100, 1)) * 1e8
    mmd = MMD(scale=1.)(x, y)
    assert np.allclose(mmd, 2.)