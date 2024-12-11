"""Generalized Bayes factors, using different divergences."""
import numpy as np
from scipy.stats import wasserstein_distance

from npp.mmd import MMD
from npp.ksd import KSD
from npp.wasserstein import Wasserstein

class gBF:
    """General class for gBFs that depend on divergences that use samples from both distributions."""
    def __init__(self, divergence, rate, spike_prob, transform=False):
        self.divergence = divergence
        self.rate = rate
        self.constant = (1/spike_prob) - 1
        if transform:
            self.transform_f = lambda x: np.exp(x - 1) * x
        else:
            self.transform_f = lambda x: x

    def gBF(self, prior_samples, posterior_samples, x):
        # posterior_samples, prior_samples
        # if divergence is two-sample: n_parameter_samples x n_likelihood_samples (array)
        # if divergence is one-sample: n_parameter_samples (list of models)
        # x: n_data
        # divergence(...): n_posterior_distributions
        mean_posterior_divergence = np.mean(self.divergence(posterior_samples, x))
        mean_prior_divergence = np.mean(self.divergence(prior_samples, x))
        if mean_prior_divergence < 0:
            print('Warning: average prior divergence is negative; consider adjusting divergence.')
            mean_prior_divergence = np.abs(mean_prior_divergence)
        n = len(x)
        gbf = 1 / (1 + self.constant * self.transform_f(
                            (mean_posterior_divergence / mean_prior_divergence) * ((n+1) ** self.rate)))
        return np.clip(gbf, 1e-6, 1-1e-6)


class gBF_wasserstein(gBF):
    def __init__(self, spike_prob, rate=0.49, wasserstein_p=1, transform=False):
        self.rate = rate
        self.spike_prob = spike_prob
        self.wasserstein_p = wasserstein_p
        self.wasserstein = Wasserstein(wasserstein_p)
        super().__init__(self._batch_wasserstein, self.rate, spike_prob, transform=transform)

    def _batch_wasserstein(self, batched_vals, compare_vals):
        batch_wass = np.array([self.wasserstein(elem, compare_vals)**self.wasserstein_p
                               for elem in batched_vals])
        return batch_wass


class gBF_MMD(gBF):
    def __init__(self, spike_prob, rate=0.49, kernel='rbf', transform=False):
        self.rate = rate
        self.spike_prob = spike_prob
        self.mmd = MMD(kernel=kernel)
        super().__init__(self._batch_mmd, self.rate, spike_prob, transform=transform)

    def _batch_mmd(self, batched_vals, compare_vals):
        """Compute MMD for a batch of datasets drawn from different models
        TODO: vectorize to accelerate (if acceptable in terms of memory).
        """
        return np.array([self.mmd(elem[:, None], compare_vals[:, None]) for elem in batched_vals])


class gBF_KSD(gBF):
    def __init__(self, spike_prob, rate=0.49, kernel='imq', transform=False):
        self.rate = rate
        self.spike_prob = spike_prob
        self.ksd = KSD(kernel=kernel)
        super().__init__(self._batch_ksd, self.rate, spike_prob, transform=transform)

    def _batch_ksd(self, batched_vals, compare_vals):
        """Compute MMD for a batch of datasets drawn from different models"""
        return np.array([self.ksd(elem, compare_vals[:, None]) for elem in batched_vals])