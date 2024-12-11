import numpy as np


class Wasserstein:
    """Wasserstein estimation for 1D empirical distributions.

    Implementation follows
    https://stackoverflow.com/a/69699278
    """
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x, y):
        # Squeeze and sort.
        x_sort = np.sort(x.squeeze())
        y_sort = np.sort(y.squeeze())
        xn, yn = len(x), len(y)

        # CDFs of each distribution, as a function of data index.
        x_cdf = np.arange(1, xn+1)/xn
        y_cdf = np.arange(1, yn+1)/yn

        # Points at which we need to evaluate the quantile function.
        xy_sort = np.sort(np.hstack((x_cdf, y_cdf)))

        # Interval sizes for integral.
        interval_sz = np.diff(np.hstack((0, xy_sort)))

        # Quantile functions
        x_quantile = x_sort[np.digitize(xy_sort, x_cdf + 1e-12, right=True)]
        y_quantile = y_sort[np.digitize(xy_sort, y_cdf + 1e-12, right=True)]

        # Approximate integral via weighted sum.
        return np.sum(interval_sz * np.abs(x_quantile - y_quantile)**self.p)**(1/self.p)

        # Compute average distance between sorted values.
        # return (np.abs(x_sort - y_sort)**self.p).mean()**(1/self.p)