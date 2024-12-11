import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def summarize_ate_samples(ate_summary, model='pm', cause_gene='TCF7', target_gene='SELL', summary='negative'):
    sample_sizes = list(ate_summary.keys())
    ate_med = [np.median(ate_summary[size]) for size in sample_sizes]
    ate_lower = [np.percentile(ate_summary[size], 2.5) for size in sample_sizes]
    ate_upper = [np.percentile(ate_summary[size], 97.5) for size in sample_sizes]

    plt.figure(figsize=(8, 6))
    plt.plot(sample_sizes, ate_med, marker='o', linestyle='-', color='blue', label="Median")
    plt.fill_between(sample_sizes, ate_lower, ate_upper, color='blue', alpha=0.2, label="95% Credible Interval")

    if summary == 'positive':
        plt.axhline(y=0.5, color='black', linestyle='--', label="0.5")
        plt.title(f"P(ATE > 0) for {cause_gene} on {target_gene}")
        plt.xlabel("n")
        plt.ylabel(f"P(ATE > 0)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/ATE_{model}_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()
    elif summary == 'negative':
        plt.title(f"P(ATE < 0) for {cause_gene} on {target_gene}")
        plt.xlabel("n")
        plt.ylabel(f"P(ATE < 0)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/ATE_{model}_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()

def plot_ate_samples(ate_summary_pm, ate_summary_np, ate_summary_gnpp,  cause_gene ='TCF7', target_gene = 'SELL',summary = 'negative'):
    sample_sizes = list(ate_summary_pm.keys())
    ate_med_pm = [np.median(ate_summary_pm[size]) for size in sample_sizes]
    ate_med_np = [np.median(ate_summary_np[size]) for size in sample_sizes]
    ate_med_gnpp = [np.median(ate_summary_gnpp[size]) for size in sample_sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, ate_med_pm, marker='o', linestyle='-', color='blue', label="parametric")
    plt.plot(sample_sizes, ate_med_np, marker='s', linestyle='-', color='green', label="nonparametric")
    plt.plot(sample_sizes, ate_med_gnpp, marker='^', linestyle='-', color='red', label="gNPP (MMD)")

    if summary == 'positive':
        plt.axhline(y=0.5, color='black', linestyle='--', label="0.5")
        plt.title(f"Median P(ATE > 0)  for {cause_gene} on on {target_gene}")
        plt.xlabel("n")
        plt.ylabel(f"P(ATE > 0)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/ATE_median_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()
    if summary == 'negative':
        plt.axhline(y=0.5, color='black', linestyle='--', label="0.5")
        plt.title(f"Median P(ATE < 0)  for {cause_gene} on {target_gene}")
        plt.xlabel("n")
        plt.ylabel(f"P(ATE < 0)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/ATE_median_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()


def positive_effect(samples):
    return (samples > 0).mean()

def negative_effect(samples):
    return (samples < 0).mean()


def save_results(cause_gene, target_gene, ate_summary_pm, ate_summary_np, ate_summary_gnpp, summary_gbf):
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Define file paths
    pm_path = f'results/{cause_gene}_{target_gene}_ate_summary_pm.pkl'
    np_path = f'results/{cause_gene}_{target_gene}_ate_summary_np.pkl'
    gnpp_path = f'results/{cause_gene}_{target_gene}_ate_summary_gnpp.pkl'
    gbf_path = f'results/{cause_gene}_{target_gene}_summary_gbf.pkl'

    # Save the results to pickle files
    with open(pm_path, 'wb') as f:
        pickle.dump(ate_summary_pm, f)

    with open(np_path, 'wb') as f:
        pickle.dump(ate_summary_np, f)

    with open(gnpp_path, 'wb') as f:
        pickle.dump(ate_summary_gnpp, f)

    with open(gbf_path, 'wb') as f:
        pickle.dump(summary_gbf, f)
