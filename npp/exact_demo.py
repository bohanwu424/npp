from aim import Figure, Distribution
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
from scipy import stats, special

import npp.PolyaTreeEmbedded as PT
from npp.manager import create_run


class GaussianGaussian:

    def __init__(self, prior_loc, prior_scale, like_scale):
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.like_scale = like_scale

        # Setup prior.
        self.prior = stats.norm(self.prior_loc, self.prior_scale)

    def prior(self):
        return self.prior

    def likelihood(self, theta):

        return stats.norm(loc=theta, scale=self.like_scale)

    def marginalln(self, x):

        # Setup.
        n = len(x)

        # Log normalizer
        lnZ = (n * np.log(self.like_scale) + np.log(self.prior_scale) + (n/2) * np.log(2 * np.pi) -
               0.5 * np.log(n/(self.like_scale**2) + 1/self.prior_scale))

        # Term in exponent.
        expTerm = 0.5 * (((np.sum(x) / (self.like_scale**2) + self.prior_loc / (self.prior_scale**2))**2)
                         / (n / (self.like_scale**2) + 1 / (self.prior_scale**2))
                         - np.sum(x**2) / (self.like_scale**2) - (self.prior_loc**2) / (self.prior_scale**2))

        return expTerm - lnZ

    def posterior(self, x):
        xsum = np.sum(x)
        xn = len(x)
        prior_var = self.prior_scale**2
        like_var = self.like_scale ** 2
        pvar = 1 / (1 / prior_var + xn / like_var)
        pmean = (self.prior_loc / prior_var + xsum / like_var) * pvar
        return stats.norm(loc=pmean, scale=np.sqrt(pvar))

    def posterior_predictive(self, x):
        xsum = np.sum(x)
        xn = len(x)
        prior_var = self.prior_scale ** 2
        like_var = self.like_scale ** 2
        postvar = 1 / (1 / prior_var + xn / like_var)
        pmean = (self.prior_loc / prior_var + xsum / like_var) * postvar
        post_predict_var = postvar + like_var
        return stats.norm(loc=pmean, scale=np.sqrt(post_predict_var))


class NPP:
    """Gaussian-Gaussian model, nonparametrically perturbed with a Polya tree."""
    def __init__(self, X, prior_loc, prior_scale, like_scale, spike_prob, h_prior_scale, h_prior_offset=0.1,
                 nsamples=100):
        # Store data.
        self.X = X

        # Setup parametric model.
        self.parametric = GaussianGaussian(prior_loc, prior_scale, like_scale)
        param_prior = self.parametric.prior
        param_like = self.parametric.likelihood
        param_posterior = self.parametric.posterior(X)
        self.parametric_predictive = self.parametric.posterior_predictive(X)

        # Setup nonparametric model.
        partition_distribution = param_like(param_posterior.mean())
        h_prior = stats.expon(loc=h_prior_offset, scale=h_prior_scale)  # ***
        self.nonparametric = PT.PolyaTreeEmbedded(param_prior, param_like, partition_distribution,
                                                  h_prior, nsamples)

        # Compute Bayes factor and posterior probability of parametric.
        importance_distribution = stats.t(40, loc=param_posterior.mean(), scale=param_posterior.std())
        self.bayesfactor_ln = self.nonparametric.bayesfactor_ln(X, importance_distribution)
        self.param_prob = special.expit(self.bayesfactor_ln + np.log(spike_prob) - np.log(1. - spike_prob))

    def posterior_predictive_ln(self, xnew):
        param_post_predict_ln = self.parametric_predictive.logpdf(xnew)
        np_post_predict_ln = self.nonparametric.posterior_predictive_ln(xnew)
        nan_frac = np.mean(np.isnan(np_post_predict_ln))

        npp_post_predict_ln = np.logaddexp(
                np.log(self.param_prob) + param_post_predict_ln,
                np.log(1 - self.param_prob) + np.nan_to_num(np_post_predict_ln, nan=-np.inf))


        return npp_post_predict_ln, param_post_predict_ln, np_post_predict_ln


def main(args):
    # Setup run tracker.
    aim_run = create_run('npp-demo', args)
    out_file = os.path.join(aim_run['local_dir'], 'results.pkl')
    print('Results: ', out_file)
    if args.log is not None:
        with open(args.log, 'a') as f:
            f.write('{},{}\n'.format(args.name, out_file))

    # Setup.
    target_mean = 0
    target_std = 1
    skew = args.skew
    ns = np.linspace(args.n_start, args.n_end, args.n_len).astype(np.int32)
    prior_loc = 0
    prior_scale = 1
    like_scale = target_std
    spike_prob = args.spike_prob
    h_prior_scale = args.h_prior_scale
    h_prior_offset = args.h_prior_offset

    # True distribution: skew normal.
    skewnorm_scale = target_std / np.sqrt(1 - 2 * (skew ** 2) / (np.pi * (1 + (skew ** 2))))
    skewnorm_loc = target_mean - skewnorm_scale * np.sqrt(2 / np.pi) * skew / np.sqrt(1 + (skew ** 2))
    p_true = stats.skewnorm(skew, loc=skewnorm_loc, scale=skewnorm_scale)
    p_true_entropy = p_true.entropy()

    # Draw true data:
    max_n = np.max(ns)
    repeats = args.repeats
    np.random.seed(seed=args.seed)
    x_all = p_true.rvs(max_n * repeats)
    x_heldout = p_true.rvs(args.nheldout)

    # Plot.
    aim_run.track(Distribution(x_all), name='data')

    # Storage.
    BFs = np.zeros((repeats, len(ns)))
    posterior_predictive_kl = np.zeros((repeats, len(ns))) - p_true_entropy
    parametric_kl = np.zeros((repeats, len(ns))) - p_true_entropy
    nonparametric_kl = np.zeros((repeats, len(ns))) - p_true_entropy
    prob_param = np.zeros((repeats, len(ns)))
    for rep in range(repeats):
        # Get independent dataset.
        x_rep = x_all[slice(max_n*rep, max_n*(rep+1))]

        # Iterate over dataset sizes.
        for j, n in enumerate(ns):
            # Subset data.
            x = x_rep[:n]

            # Setup NPP.
            npp = NPP(x, prior_loc, prior_scale, like_scale, spike_prob, h_prior_scale, h_prior_offset,
                      nsamples=args.nsamples)
            BFs[rep, j] = npp.bayesfactor_ln
            prob_param[rep, j] = npp.param_prob

            # Computer posterior predictive log likelihood.
            npp_pll, p_pll, np_pll = npp.posterior_predictive_ln(x_heldout)
            posterior_predictive_kl[rep, j] -= np.mean(npp_pll)
            parametric_kl[rep, j] -= np.mean(p_pll)
            nonparametric_kl[rep, j] -= np.mean(np_pll)  # ** [~np.isnan(np_pll)]

            # Track results.
            aim_run.track({'n': n, 'BF': BFs[rep, j], 'prob_param': prob_param[rep, j],
                           'posterior_predict_kl': posterior_predictive_kl[rep, j]})

        aim_run.track({'repeat': rep})

    # Save results
    with open(out_file, 'wb') as f:
        pickle.dump({
            'ns': ns,
            'BFs': BFs,
            'NPP_kl': posterior_predictive_kl,
            'parametric_kl': parametric_kl,
            'nonparametric_kl': nonparametric_kl,
            'prob_param': prob_param
        }, f)

    # Summarize results
    results = pd.DataFrame({'n': ns, 'BF_mean': np.mean(BFs, axis=0), 'BF_std': np.std(BFs, axis=0),
                            'BF_se': np.std(BFs, axis=0)/np.sqrt(repeats),
                            'prob_param_mean': np.mean(prob_param, axis=0),
                            'prob_param_std': np.std(prob_param, axis=0),
                            'prob_param_se': np.std(prob_param, axis=0)/np.sqrt(repeats),
                            'posterior_predictive_kl_mean': np.mean(posterior_predictive_kl, axis=0),
                            'posterior_predictive_kl_std': np.std(posterior_predictive_kl, axis=0),
                            'posterior_predictive_kl_se': np.std(posterior_predictive_kl, axis=0) / np.sqrt(repeats),
                            'parametric_kl_mean': np.mean(parametric_kl, axis=0),
                            'parametric_kl_std': np.std(parametric_kl, axis=0),
                            'parametric_kl_se': np.std(parametric_kl, axis=0) / np.sqrt(repeats),
                            'nonparametric_kl_mean': np.mean(nonparametric_kl, axis=0),
                            'nonparametric_kl_std': np.std(nonparametric_kl, axis=0),
                            'nonparametric_kl_se': np.std(nonparametric_kl, axis=0) / np.sqrt(repeats)
                            })
    bf_fig = px.line(data_frame=results, x='n', y='BF_mean', error_y='BF_se',
                     labels={'x': 'n', 'y': 'log Bayes factor: parametric/nonparametric'})
    aim_run.track(Figure(bf_fig), name='bayes_factor')
    ppa_fig = px.line(data_frame=results, x='n', y='prob_param_mean', error_y='prob_param_se',
                     labels={'x': 'n', 'y': 'Posterior prob. of parametric model'})
    aim_run.track(Figure(ppa_fig), name='prob_parametric')
    long_results = results.melt(id_vars='n', value_vars=['posterior_predictive_kl_mean', 'parametric_kl_mean', 'nonparametric_kl_mean'],
                                value_name='kl', var_name='model')
    long_results['kl_error'] = results[['posterior_predictive_kl_se', 'parametric_kl_se', 'nonparametric_kl_se']
                                       ].unstack().values
    pkl_fig = px.line(data_frame=long_results, x='n', y='kl', error_y='kl_error', color='model')
    aim_run.track(Figure(pkl_fig), name='posterior_predictive_kl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPP toy example with a Polya Tree')
    parser.add_argument('--log', default=None, help='File in which to log output results.')
    parser.add_argument('--name', help='Name to give results in log file.')
    parser.add_argument('--skew', default=0., type=float, help='Skew of p_0 (a skew normal distribution).')
    parser.add_argument('--n-start', default=5, type=int, help='Smallest data size considered')
    parser.add_argument('--n-end', default=20, type=int, help='Largest data size considered')
    parser.add_argument('--n-len', default=3, type=int, help='Number of data sizes considered.')
    parser.add_argument('--spike-prob', default=0.5, type=float, help='Probability of spike (parametric model).')
    parser.add_argument('--h-prior-scale', default=100., type=float, help='Scale of exponential prior on h.')
    parser.add_argument('--h-prior-offset', default=0.1, type=float, help='Location (left support) of exponential prior on h.')
    parser.add_argument('--repeats', default=1, type=int, help='Number of times to repeat the experiment.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for data.')
    parser.add_argument('--nsamples', default=100, type=int, help='Number of Monte Carlo samples for Bayes factor.')
    parser.add_argument('--nheldout', default=1000, type=int, help='Number of Monte Carlo samples for KL.')
    args0 = parser.parse_args()

    main(args0)
