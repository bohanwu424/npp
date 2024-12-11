import numpy as np
from scipy.special import logsumexp


class PTNode:
    """Each node represents an interval."""
    def __init__(self, eps, iv, lp, Flp_iv):
        # eps -- index of node in tree (a 1D array of 1s and 0s). epsilon in B&G notation
        self.eps = eps
        self.m_eps = len(eps)
        # iv -- interval covered by node. B_eps in B&G notation
        self.iv = iv
        # lp -- log probability of interval (array of size L x L, where L is number of Monte Carlo samples)
        # mu(B_eps) in B&G notation
        self.lp = lp
        # Flp_iv - value of F at the interval endpoints. array of size 2 x L
        self.Flp_iv = Flp_iv
        # n - Number of points that fall in this interval.
        self.n = 0
        # If the node is a terminal node, and there is only one value of x in the interval,
        # we store that value of x.
        self.x = None
        # Initialize storage for child nodes.
        self.kids = []

    def add(self, xnew, Finv, flp, Flp, hln, update=True):
        # Note f and F will in general output an L vector.
        if self.n == 0:
            # If no other points fall in this node's interval, it is a terminal node.
            if update:
                # We store xnew if xnew is part of the dataset (but not if it is just used for evaluation).
                self.n = 1
                self.x = xnew
            return 0.

        if len(self.kids) == 0:
            # In this case, the node is terminal in the tree data structure.
            # We must initialize its children, and move any point that used to terminate here
            # to the corresponding child.

            # Compute midpoint of the interval.
            self.split = Finv(np.sum(self.eps / (2 ** (1 + np.arange(self.m_eps))))
                              + 1/(2 ** (self.m_eps + 1)))
            # Compute rho factor (d(epsilon) in B&G notation)
            self.rho_ln = 2 * (flp(self.split) - self.lp) - hln
            # Compute log probability (mu(B)) for sub-intervals (children).
            Flp_mid = Flp(self.split)
            child_ivs = [[self.iv[0], self.split], [self.split, self.iv[1]]]
            Flp_ivs = np.array([[self.Flp_iv[0], Flp_mid], [Flp_mid, self.Flp_iv[1]]])
            # This subtraction operation is sometimes not numerically stable, so clipped.
            child_lps = logsumexp(Flp_ivs, axis=1, b=np.array([-1., 1.])[None, :, None])
            # Store alpha, the concentrations of the Beta distribution for this node.
            self.alpha_ln = np.array([self.rho_ln + 0.5 * (child_lps[0] - child_lps[1]),
                                      self.rho_ln + 0.5 * (child_lps[1] - child_lps[0])])
            self.alpha_sum_ln = logsumexp(self.alpha_ln, axis=0)
            # Initialize child nodes.
            for k in range(2):
                self.kids.append(PTNode(np.concatenate([self.eps, [k]]), child_ivs[k],
                                        child_lps[k], Flp_ivs[k]))
            # Currently, this node is terminal for self.x. We need to make the corresponding child node
            # terminal for self.x.
            x_child = int(self.x > self.split)
            self.kids[x_child].n = 1
            self.kids[x_child].x = self.x
            self.x = None

        # Compute the factor contributing to the log marginal likelihood (g_j, within psi in B&G notation)
        log_gnew = self.alpha_sum_ln - np.logaddexp(self.alpha_sum_ln, np.log(self.kids[0].n + self.kids[1].n))
        xnew_child = int(xnew > self.split)
        log_gnew += (np.logaddexp(self.alpha_ln[xnew_child], np.log(self.kids[xnew_child].n + 1e-32)) -
                     self.alpha_ln[xnew_child])

        if update:
            # Update counts.
            self.n += 1

        # Add to next node.
        return log_gnew + self.kids[xnew_child].add(xnew, Finv, flp, Flp, hln, update=update)


class PolyaTreeEmbedded:

    def __init__(self, param_prior, param_like, partition_distribution,
                 h_distribution, nsamples, support=(-np.inf, np.inf)):
        # Store embedded parametric model.
        self.param_prior = param_prior
        self.param_like = param_like
        self.partition_distribution = partition_distribution
        # Store distribution over h.
        self.h_distribution = h_distribution
        # Initialize root node of tree.
        self.root = PTNode([], support, np.zeros(nsamples),
                           np.concatenate([np.zeros(nsamples)[None, :] - np.inf,
                                           np.zeros(nsamples)[None, :]], axis=0))

        # Number of samples.
        self.nsamples = nsamples

    def theta_weights(self, x, importance_distribution):
        # Draw samples from importance distribution over theta.
        thetas = importance_distribution.rvs(size=self.nsamples)
        # Compute unnormalized weight: the joint log probability of theta and x,
        # minus the log probability under the importance distribution
        weight_un = (np.sum(self.param_like(thetas[:, None]).logpdf(x[None, :]), axis=1)
                     + self.param_prior.logpdf(thetas)
                     - importance_distribution.logpdf(thetas))
        weightsln = weight_un - logsumexp(weight_un)
        return thetas, weightsln

    def bayesfactor_ln(self, x, importance_distribution):
        # Bayes factor: log marginal likelihood of parametric model minus log marginal likelihood of PT model.

        # Get thetas (parametric model parameter samples) and their importance weights.
        self.thetas, self.weightsln = self.theta_weights(x, importance_distribution)

        # Get parametric model functions, conditioned on theta.
        self.Finv = self.partition_distribution.ppf
        self.flp = self.param_like(self.thetas).logpdf
        self.Flp = self.param_like(self.thetas).logcdf

        # Draw samples from h distribution.
        hs = self.h_distribution.rvs(size=self.nsamples)
        self.hln = np.log(hs)

        log_psi = 0.
        for j in range(len(x)):
            log_psi += self.root.add(x[j], self.Finv, self.flp, self.Flp, self.hln, update=True)

        self.log_psi_weighted = log_psi + self.weightsln

        self.bayesfactor_ln = -logsumexp(self.log_psi_weighted)
        return self.bayesfactor_ln

    def posterior_predictive_ln(self, xout):
        log_psi_out = np.zeros((self.nsamples, len(xout)))
        for j in range(len(xout)):
            log_psi_out[:, j] = self.root.add(xout[j], self.Finv, self.flp, self.Flp, self.hln, update=False)

        log_param_like = self.param_like(self.thetas[:, None]).logpdf(xout[None, :])
        log_predictive = (logsumexp(log_param_like + log_psi_out + self.log_psi_weighted[:, None], axis=0) +
                          self.bayesfactor_ln)
        if np.any(np.isinf(log_predictive)):
            import pdb
            pdb.set_trace()
        return log_predictive
