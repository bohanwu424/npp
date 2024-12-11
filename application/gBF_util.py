import numpy as np
from npp.mmd import MMD
from npp.generalized_BFs import gBF

# Function to sample from the prior
def sample_prior_and_posterior(X, a, y, n_samples=1000, n_y=100):
    n_features = X.shape[1]
    n_obs = len(a)

    sigma_prior = np.abs(np.random.standard_cauchy(size=n_y))
    mu_prior = np.random.uniform(-100, 100, size=n_y)
    beta_prior = np.random.uniform(-100, 100, size=(n_y, n_features))
    tau_prior = np.random.uniform(-100, 100, size=n_y)

    dirichlet_weights = np.random.dirichlet(alpha=np.ones(n_obs), size=n_samples)
    prior_samples = []
    post_samples = []

    for i in range(n_samples):
        sampled_indices = np.random.choice(np.arange(n_obs), size=n_obs, p=dirichlet_weights[i], replace=True)
        sampled_X = X[sampled_indices]
        sampled_a = a[sampled_indices]
        sampled_y = y[sampled_indices]

        # Sample from prior
        for j in range(n_y):
            y_mean_prior = mu_prior[j] + np.dot(sampled_X, beta_prior[j]) + tau_prior[j] * sampled_a
            y_samples_prior = np.random.normal(loc=y_mean_prior, scale=sigma_prior[j])
            y_samples_prior = y_samples_prior[:, np.newaxis]
            prior_samples.append(np.hstack([sampled_X, sampled_a[:, np.newaxis], y_samples_prior]))

        # Sample from posterior
        X_design = np.hstack([np.ones((len(a), 1)), sampled_X, sampled_a[:, np.newaxis]])
        XtX_inv = np.linalg.inv(X_design.T @ X_design + np.eye(X_design.shape[1]) * 1e-6)
        beta = XtX_inv @ X_design.T @ sampled_y
        sigma2 = np.sum((sampled_y - X_design @ beta) ** 2) / (n_obs - 1)
        beta_samples = np.random.multivariate_normal(mean=beta, cov=XtX_inv, size=n_y)
        for j in range(n_y):
            y_mean_post = np.dot(X_design, beta_samples[j])
            y_samples_post = np.random.normal(loc=y_mean_post, scale=np.sqrt(sigma2))
            post_samples.append(np.hstack([sampled_X, sampled_a[:, np.newaxis], y_samples_post[:, np.newaxis]]))

    return prior_samples, post_samples

# Class to compute gBF using MMD
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
        return np.array([self.mmd(elem, compare_vals) for elem in batched_vals])

# Compute gBayesFactor
def compute_gBayesFactor(X, a, y, n_post=100, n_y=10, spike_prob=0.5, rate=0.49, kernel='imq', transform=False, n_subsample=100):
    gbf_mmd = gBF_MMD(spike_prob, rate, kernel, transform)

    prior_samples, posterior_samples = sample_prior_and_posterior(X, a, y, n_samples=n_post, n_y=n_y)

    if n_subsample is not None:
        idx = np.random.choice(np.arange(len(prior_samples)), size=n_subsample, replace=False)
        prior_samples = [prior_samples[i] for i in idx]
        posterior_samples = [posterior_samples[i] for i in idx]

    data = np.hstack([X, a[:, np.newaxis], y[:, np.newaxis]])
    gBayesFactor_value = gbf_mmd.gBF(prior_samples, posterior_samples, data)

    del prior_samples, posterior_samples, data
    return gBayesFactor_value