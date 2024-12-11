from aim import Figure, Distribution
import argparse
import numpy as np
import os
import plotly.express as px
import pandas as pd
import pickle
from scipy import stats

from npp.exact_demo import GaussianGaussian
from npp.generalized_BFs import gBF_wasserstein, gBF_MMD, gBF_KSD
from npp.manager import create_run


def weighted_quantile(quant, values, weights):
    # Get median of a weighted empirical distribution.
    # Based roughly on https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy.
    # values: size n
    # weights: size n x k

    # Sort values
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    # Values gives the value of each quantile defined below; the second term puts these in the center of each interval.
    weighted_quantiles = np.cumsum(weights, axis=0) - 0.5 * weights

    return np.array([np.interp(quant, weighted_quantiles[:, j], values) for j in range(weights.shape[1])])


class gNPP:
    """Gaussian-Gaussian model, nonparametrically perturbed with a DP, using a generalized Bayes factor."""

    def __init__(self, prior_loc, prior_scale, like_scale, spike_prob, rate=0.49, gBF_type='wass',
                 parameter_samples=100, likelihood_samples=1000, CI=90, kernel='rbf', transform=False):

        # Setup generalized Bayes factor.
        self.gBF_type = gBF_type
        if gBF_type == 'wass':
            self.gBF = gBF_wasserstein(spike_prob, rate, transform=transform)
        elif gBF_type == 'mmd':
            self.gBF = gBF_MMD(spike_prob, rate, transform=transform)
        elif gBF_type == 'ksd':
            self.gBF = gBF_KSD(spike_prob, rate, transform=transform)
        else:
            assert False, gBF_type + ' not yet implemented'

        # Setup parametric model.
        self.parametric = GaussianGaussian(prior_loc, prior_scale, like_scale)
        # Draw samples from parametric model prior.
        prior_theta_samps = self.parametric.prior.rvs(parameter_samples,)
        if gBF_type in ['wass', 'mmd']:
            # These divergences need samples from the parametric model prior.
            self.prior_samps = self.parametric.likelihood(prior_theta_samps[:, None]
                                                          ).rvs((parameter_samples, likelihood_samples))
        elif gBF_type == 'ksd':
            # These divergences do not need samples from the parametric model posterior.
            self.prior_samps = [self.parametric.likelihood(theta_samp)
                                for theta_samp in prior_theta_samps[:, None]]

        # Number of Monte Carlo samples.
        self.parameter_samples = parameter_samples
        self.likelihood_samples = likelihood_samples
        # Credible interval lower quantile.
        self.ci_quant = (100-CI)/200

    def posterior_median(self, x):
        parameter_samples = self.parameter_samples
        likelihood_samples = self.likelihood_samples

        # Get parametric model posterior samples of median (=mean for Gaussian).
        parametric_posterior = self.parametric.posterior(x)
        parametric_theta_samps = parametric_posterior.rvs(parameter_samples,)
        # Summarize parametric posterior.
        parametric_CI = (np.quantile(parametric_theta_samps, self.ci_quant),
                         np.quantile(parametric_theta_samps, 1 - self.ci_quant))
        parametric_mean = np.mean(parametric_theta_samps)

        # Compute generalized Bayes factor.
        if self.gBF_type in ['wass', 'mmd']:
            # These divergences need samples from the parametric model posterior.
            parametric_samps = self.parametric.likelihood(parametric_theta_samps[:, None]
                                                          ).rvs((parameter_samples, likelihood_samples))
            gBayesFactor = self.gBF.gBF(self.prior_samps, parametric_samps, x)
        elif self.gBF_type in ['ksd']:
            # These divergences do not need samples from the parametric model posterior.
            parametric_samps = [self.parametric.likelihood(theta_samp)
                                for theta_samp in parametric_theta_samps[:, None]]
            gBayesFactor = self.gBF.gBF(self.prior_samps, parametric_samps, x)

        # Get nonparametric model posterior samples of median.
        bayes_boot_weights = np.random.dirichlet(np.ones(len(x)), size=parameter_samples).T
        nonparametric_theta_samps = weighted_quantile(0.5, x, bayes_boot_weights)
        # Summarize nonparametric posterior.
        nonparametric_CI = (np.quantile(nonparametric_theta_samps, self.ci_quant),
                            np.quantile(nonparametric_theta_samps, 1 - self.ci_quant))
        nonparametric_mean = np.mean(nonparametric_theta_samps)

        # Summarize gNPP posterior.
        gnpp_mean = gBayesFactor * parametric_mean + (1 - gBayesFactor) * nonparametric_mean
        combined_samps = np.concatenate([parametric_theta_samps, nonparametric_theta_samps])
        combined_weights = np.concatenate([gBayesFactor * np.ones(parameter_samples) / parameter_samples,
                                           (1 - gBayesFactor) * np.ones(parameter_samples) / parameter_samples])
        gnpp_CI = (weighted_quantile(self.ci_quant, combined_samps, combined_weights[:, None])[0],
                   weighted_quantile(1 - self.ci_quant, combined_samps, combined_weights[:, None])[0])

        return gnpp_mean, gnpp_CI, parametric_mean, parametric_CI, nonparametric_mean, nonparametric_CI, gBayesFactor


def main(args):
    # Setup run tracker.
    aim_run = create_run('gnpp-demo', args)
    out_file = os.path.join(aim_run['local_dir'], 'results.pkl')
    print('Results: ', out_file)
    if args.log is not None:
        with open(args.log, 'a') as f:
            f.write('{},{}\n'.format(args.name, out_file))

    # Setup.
    target_mean = 0
    target_std = 1
    skew = args.skew
    n_len = args.n_len
    ns = np.linspace(args.n_start, args.n_end, n_len).astype(np.int32)
    prior_loc = 0
    prior_scale = 1
    like_scale = target_std
    spike_prob = args.spike_prob
    divergence = args.diverge
    rate = args.rate
    target_coverage = 90

    # True distribution: skew normal.
    skewnorm_scale = target_std / np.sqrt(1 - 2 * (skew ** 2) / (np.pi * (1 + (skew ** 2))))
    skewnorm_loc = target_mean - skewnorm_scale * np.sqrt(2 / np.pi) * skew / np.sqrt(1 + (skew ** 2))
    p_true = stats.skewnorm(skew, loc=skewnorm_loc, scale=skewnorm_scale)
    p_true_median = p_true.median()

    # Draw true data:
    max_n = np.max(ns)
    repeats = args.repeats
    np.random.seed(seed=args.seed)
    x_all = p_true.rvs(max_n * repeats)

    # Plot.
    aim_run.track(Distribution(x_all), name='data')

    # Storage.
    gBFs = np.zeros((repeats, n_len))
    parametric_point_err = np.zeros((repeats, n_len))
    nonparametric_point_err = np.zeros((repeats, n_len))
    gNPP_point_err = np.zeros((repeats, n_len))
    parametric_coverage = np.zeros(n_len)
    parametric_interval_len = np.zeros((repeats, n_len))
    nonparametric_coverage = np.zeros(n_len)
    nonparametric_interval_len = np.zeros((repeats, n_len))
    gNPP_coverage = np.zeros(n_len)
    gNPP_interval_len = np.zeros((repeats, n_len))

    # Iterate over independent datasets.
    for rep in range(repeats):
        # Get dataset.
        x_rep = x_all[slice(max_n * rep, max_n * (rep + 1))]

        # Iterate over dataset sizes.
        for j, n in enumerate(ns):
            # Subset data.
            x = x_rep[:n]

            # Set number of samples for divergence.
            if args.likelihood_samples == 'match':
                likelihood_samples = n
            else:
                likelihood_samples = int(args.likelihood_samples)

            # Compute gNPP posterior, as well as component posteriors.
            gnpp = gNPP(prior_loc, prior_scale, like_scale, spike_prob, rate=rate, gBF_type=divergence,
                        parameter_samples=args.parameter_samples, likelihood_samples=likelihood_samples,
                        kernel=args.kernel, CI=target_coverage, transform=args.transform)
            gnpp_mean, gnpp_CI, parametric_mean, parametric_CI, nonparametric_mean, nonparametric_CI, gBFs[rep, j] = (
                            gnpp.posterior_median(x))

            # Record performance.
            gNPP_point_err[rep, j] = np.abs(gnpp_mean - p_true_median)
            parametric_point_err[rep, j] = np.abs(parametric_mean - p_true_median)
            nonparametric_point_err[rep, j] = np.abs(nonparametric_mean - p_true_median)
            gNPP_coverage[j] += int((p_true_median > gnpp_CI[0]) and (p_true_median < gnpp_CI[1]))/repeats
            parametric_coverage[j] += int((p_true_median > parametric_CI[0]) and (p_true_median < parametric_CI[1]))/repeats
            nonparametric_coverage[j] += int((p_true_median > nonparametric_CI[0]) and (p_true_median < nonparametric_CI[1]))/repeats
            gNPP_interval_len[rep, j] = gnpp_CI[1] - gnpp_CI[0]
            parametric_interval_len[rep, j] = parametric_CI[1] - parametric_CI[0]
            nonparametric_interval_len[rep, j] = nonparametric_CI[1] - nonparametric_CI[0]

            aim_run.track({'n': n,
                           'gNPP_point_err': gNPP_point_err[rep, j],
                           'parametric_point_err': parametric_point_err[rep, j],
                           'nonparametric_point_err': nonparametric_point_err[rep, j]})
        aim_run.track({'repeat': rep})

    # Save results.
    with open(out_file, 'wb') as f:
        pickle.dump({
                     'target_coverage': target_coverage,
                     'ns': ns,
                     'gBFs': gBFs,
                     'parametric_point_err': parametric_point_err,
                     'nonparametric_point_err': nonparametric_point_err,
                     'gNPP_point_err': gNPP_point_err,
                     'parametric_coverage': parametric_coverage,
                     'parametric_interval_len': parametric_interval_len,
                     'nonparametric_coverage': nonparametric_coverage,
                     'nonparametric_interval_len': nonparametric_interval_len,
                     'gNPP_coverage': gNPP_coverage,
                     'gNPP_interval_len': gNPP_interval_len}, f)

    # Summarize results.
    results = pd.DataFrame({
        'n': ns,
        'gBF_mean': np.mean(gBFs, axis=0), 'gBF_std': np.std(gBFs, axis=0),
        'gBF_se': np.std(gBFs, axis=0) / np.sqrt(repeats),
        'gnpp_rmse': np.sqrt(np.mean(gNPP_point_err**2, axis=0)),
        'parametric_rmse': np.sqrt(np.mean(parametric_point_err**2, axis=0)),
        'nonparametric_rmse': np.sqrt(np.mean(nonparametric_point_err**2, axis=0)),
        'gnpp_coverage': gNPP_coverage,
        'parametric_coverage': parametric_coverage,
        'nonparametric_coverage': nonparametric_coverage,
        'gnpp_interval_mn': np.mean(gNPP_interval_len, axis=0),
        'parametric_interval_mn': np.mean(parametric_interval_len, axis=0),
        'nonparametric_interval_mn': np.mean(nonparametric_interval_len, axis=0)
    })
    bf_fig = px.line(data_frame=results, x='n', y='gBF_mean', error_y='gBF_se',
                     labels={'x': 'n', 'y': 'generalized Bayes factor: parametric/nonparametric'})
    aim_run.track(Figure(bf_fig), name='gbayes_factor')

    long_results = results.melt(id_vars='n', value_vars=['gnpp_rmse', 'parametric_rmse', 'nonparametric_rmse'],
                                value_name='rmse', var_name='model')
    point_rmse_fig = px.line(data_frame=long_results, x='n', y='rmse', color='model', log_y=True)
    aim_run.track(Figure(point_rmse_fig), name='point_estimate_rmse')

    long_results = results.melt(id_vars='n', value_vars=['gnpp_coverage', 'parametric_coverage', 'nonparametric_coverage'],
                                value_name='coverage', var_name='model')
    coverage_fig = px.line(data_frame=long_results, x='n', y='coverage', color='model')
    aim_run.track(Figure(coverage_fig), name='interval_coverage')

    long_results = results.melt(id_vars='n',
                                value_vars=['gnpp_interval_mn', 'parametric_interval_mn', 'nonparametric_interval_mn'],
                                value_name='interval_len', var_name='model')
    interval_len_fig = px.line(data_frame=long_results, x='n', y='interval_len', color='model', log_y=True)
    aim_run.track(Figure(interval_len_fig), name='interval_len')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gNPP toy example.')
    parser.add_argument('--log', default=None, help='File in which to log output results.')
    parser.add_argument('--name', help='Name to give results in log file.')
    parser.add_argument('--skew', default=0., type=float, help='Skew of p_0 (a skew normal distribution).')
    parser.add_argument('--n-start', default=5, type=int, help='Smallest data size considered')
    parser.add_argument('--n-end', default=20, type=int, help='Largest data size considered')
    parser.add_argument('--n-len', default=3, type=int, help='Number of data sizes considered.')
    parser.add_argument('--spike-prob', default=0.5, type=float,
                        help='Probability of spike (parametric model).')
    parser.add_argument('--repeats', default=1, type=int,
                        help='Number of times to repeat the experiment.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for data.')
    parser.add_argument('--parameter-samples', default=10, type=int,
                        help='Number of parameter samples for estimating gBayes factor.')
    parser.add_argument('--likelihood-samples', default='match',
                        help='Number of likelihood samples for estimating gBayes factor. Either: "match" OR int')
    parser.add_argument('--diverge', default='wass', type=str,
                        help='Divergence to use for generalized Bayes factor: wass OR mmd OR ksd.')
    parser.add_argument('--rate', default=0.49, type=float,
                        help='Set generalized Bayes factor scaling rate')
    parser.add_argument('--kernel', default='rbf', type=str,
                        help='Kernel type for kernel-based discrepancies (rbf OR imq)')
    parser.add_argument('--wass-p', default=2, type=int,
                        help='Value of p for the Wasserstein divergence (if diverge=wass)')
    parser.add_argument('--transform', action='store_true',
                        help='Transform (exponentiate) the divergence terms in the Bayes factor.')
    args0 = parser.parse_args()

    main(args0)