from numpy.random import dirichlet

from transformer import test_run
from modeling_util import subsample_ate
from misc_util import negative_effect, summarize_ate_samples, plot_ate_samples

import matplotlib.pyplot as plt
import numpy as np
import arviz as az

def run_example(cause_gene, target_gene):
    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)

    # Load data for the given gene pair
    a, y, testX_hat, testX = test_run(cause_gene, target_gene)
    n_post = 1000

    # Perform subsampling for ATE and GBF
    ate_summary_pm, ate_summary_np, ate_summary_gnpp, summary_gbf = subsample_ate(
        a, testX_hat, testX, y, negative_effect, n_post, start_size=50, n_SS=10
    )


    # Summarize and plot ATE samples
    summaries = [ate_summary_pm, ate_summary_np, ate_summary_gnpp]
    labels = ["pm", "np", "gnpp"]
    for summary, label in zip(summaries, labels):
        summarize_ate_samples(summary, label, cause_gene, target_gene)
    plot_ate_samples(*summaries, cause_gene, target_gene, summary='negative')

    # Generalized Bayes Factor (GBF) Plot
    sample_sizes = sorted(summary_gbf.keys())
    median_gbf = [np.median(summary_gbf[size]) for size in sample_sizes]
    lower_ci = [np.percentile(summary_gbf[size], 2.5) for size in sample_sizes]
    upper_ci = [np.percentile(summary_gbf[size], 97.5) for size in sample_sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, median_gbf, marker='o', linestyle='-', color='blue', label='Median')
    plt.fill_between(sample_sizes, lower_ci, upper_ci, color='blue', alpha=0.2, label='95% CI')
    plt.title(f'Generalized Bayes Factor: {cause_gene} on {target_gene}')
    plt.xlabel('Sample Size')
    plt.ylabel('GBF Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/gbf_{cause_gene}_{target_gene}.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_example(cause_gene = 'FOXP3', target_gene = 'NKG7')