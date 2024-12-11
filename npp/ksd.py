
import numpy as np
from scipy import stats


def log_score(dist, x):
    """log Stein score
    dist: distribution
    x: data"""
    if dist.dist.name == 'norm':
        return (dist.mean() - x) / dist.var()
    else:
        assert False, 'Stein score not yet implemented for ' + dist.name


class IMQ:
    def __init__(self, offset, exponent):
        self.offset2 = offset**2
        self.exponent = exponent

    def __call__(self, diff, dist2=None):
        """The IMQ kernel.
        input:
        diff: x - y values, size: N x N x D
        dist2: optionally provide precomputed squared distances ||x-y||_2^2, size: N x N
        output:
        kernel values, size N x N
        grad_x kernel values, size: N x N x D
        trace [grad_x grad_y kernel] values, size: N x N
        """
        # Setup.
        offset2, exponent = self.offset2, self.exponent
        D = diff.shape[2]

        # Compute distance squared between points.
        if dist2 is None:
            dist2 = (diff**2).sum(axis=2)

        # Kernel (N x N)
        K = (offset2 + dist2) ** self.exponent
        # Gradient_x of kernel (N x N x D).
        gradx_K = 2 * exponent * ((offset2 + dist2) ** (exponent-1))[:, :, None] * diff
        # trace [grad_x grad_y kernel] (N x N)
        tr_gradxy_K = -2 * exponent * ((2 * exponent + D - 2) * dist2 + offset2 * D
                                       ) * ((offset2 + dist2) ** (exponent-2))

        return K, gradx_K, tr_gradxy_K


def U_stat(M):
    """Compute U statistic mean from matrix of values."""
    N = M.shape[0]
    return (M.sum() - M.diagonal().sum()) / (N * (N - 1))

class KSD:
    """Kernelized Stein discrepancy"""
    def __init__(self, kernel='imq', scale='median'):
        self.kernel = kernel
        self.scale = scale

    def __call__(self, model, X):
        """The KSD.
        input: data, size: N x D
        output: KSD, size: scalar
        """

        # Model Stein score (size: N x D).
        score = log_score(model, X)

        # Data difference and distance^2
        N, D = X.shape
        diff = X[:, None, :] - X[None, :, :]
        dist2 = (diff**2).sum(axis=2)

        # Obtain scale.
        if type(self.scale) is float:
            scale = self.scale
        elif self.scale == 'median':
            # Correction gives median of off-diagonal entries (since diagonal entries are zero).
            scale = np.quantile(np.sqrt(dist2), np.minimum(0.5 + 1 / N, 1.)) + 1e-6
        else:
            assert False, 'Scale of ' + self.scale + ' is not supported.'

        # Initialize kernel.
        if self.kernel == 'imq':
            kernel = IMQ(scale, -0.5)
        else:
            assert False, self.kernel + ' kernel is not supported.'

        # Compute kernel terms.
        K, gradx_K, tr_gradxy_K = kernel(diff, dist2)

        # Compute KSD terms.
        E_score2_k = U_stat(np.einsum('id,jd,ij->ij', score, score, K))
        E_score_gradk = U_stat(np.einsum('ijd,jd->ij', gradx_K, score))
        E_gradxyk = U_stat(tr_gradxy_K)

        return E_score2_k + 2 * E_score_gradk + E_gradxyk
