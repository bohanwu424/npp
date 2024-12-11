import numpy as np


class MMD:
    """Maximum mean discrepancy (MMD) estimation"""
    def __init__(self, kernel='rbf', scale='median'):
        self.kernel = kernel
        self.scale = scale

    def _dist_2norm(self, x):
        """Compute Euclidean distance matrix between points.
        x is assumed to be a numpy array of size num_datapoints x num_features.
        """
        return np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))

    def _rbf(self, dist, scale):
        """Compute RBF kernel.
        """
        return np.exp(-0.5 * (1 / scale ** 2) * dist ** 2)

    def _imq(self, dist, scale):
        """Compute IMQ kernel."""
        return (scale**2/4 + dist**2) ** -0.5

    def _expected_kernel_U(self, K):
        """The expected value of the kernel, using a U statistic"""
        return np.sum(K - np.diag(np.diag(K))) / (K.shape[0] * (K.shape[0] - 1))

    def __call__(self, x, y):
        """Compute MMD^2."""
        x_sz, y_sz = x.shape[0], y.shape[0]
        data = np.concatenate([x, y], axis=0)

        # Compute distances.
        dist = self._dist_2norm(data)

        # Obtain scale.
        if type(self.scale) is float:
            scale = self.scale
        elif self.scale == 'median':
            # Correction gives median of off-diagonal entries (since diagonal entries are zero).
            scale = np.quantile(dist, np.minimum(0.5 + 1/(x_sz + y_sz), 1.)) + 1e-6
        else:
            assert False, 'Scale of ' + self.scale + ' is not supported.'

        # Compute kernel.
        if self.kernel == 'rbf':
            K = self._rbf(dist, scale)
        elif self.kernel == 'imq':
            K = self._imq(dist, scale)
        else:
            assert False, self.kernel + ' kernel is not supported.'

        # Compute MMD.
        Ekxx = self._expected_kernel_U(K[:x_sz][:, :x_sz])
        Ekyy = self._expected_kernel_U(K[x_sz:][:, x_sz:])
        Ekxy = (1 / (x_sz * y_sz)) * np.sum(K[:x_sz][:, x_sz:])
        return Ekxx + Ekyy - 2 * Ekxy
