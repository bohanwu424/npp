import pymc as pm
import pymc_bart as pmb
from numpy.random import dirichlet
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer

from sklearn.linear_model import LinearRegression
from statsmodels.api import add_constant, OLS

from gBF_util import compute_gBayesFactor


# modeling prop score with NN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
def prop_score(X, a, epochs=500, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    a_tensor = torch.tensor(a, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, a_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NN(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_X, batch_a in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_a)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_a = model(X_tensor).numpy()

    return predicted_a

# nonparametric model
def causal_BART(X, a, y, q98_a, q0_a, cause_gene ="FOXP3",target_gene = "NKG7", n_post=1000, n_samples_per_trace=30, n_BB=1000, n_tunes=1000, posterior_predictive_check=False, trace_plot=False, convergence_check=False):
    # Remove columns with all zeros
    X = X[:, ~(X == 0).all(axis=0)]

    # Estimate propensity score
    a_hat = prop_score(X, a)

    # Apply PowerTransformer (Yeo-Johnson) to y
    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(y.reshape(-1, 1)).flatten()


    with pm.Model() as model:
        data_X = pm.Data("df_X", np.hstack([X, a_hat]))
        data_a = pm.Data("a", a.flatten())
        mu = pmb.BART('mu', X=data_X, Y=y_transformed, m=50)
        tau = pm.Flat('tau')
        f = pm.Deterministic('f', mu + tau * data_a)
        sigma = pm.HalfNormal('sigma', sigma=1)
        y_obs = pm.Normal('y_obs', mu=f, sigma=sigma, observed=y_transformed)
        trace = pm.sample(chains=4, cores=4, tune=n_tunes, draws=2000, discard_tuned_samples=True)

    if posterior_predictive_check:
        with model:
            ppm = pm.sample_posterior_predictive(trace)
        y_pred_transformed = ppm.posterior_predictive['y_obs'].isel(chain=-1, draw=-1).values
        y_pred = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
        plt.hist(y, bins=30, alpha=0.5, label="Observed y")
        plt.hist(y_pred, bins=30, alpha=0.5, label="Posterior Predictive y")
        plt.legend()
        plt.xlabel("y")
        plt.ylabel("Frequency")
        plt.title("Posterior Predictive Check")
        plt.tight_layout()
        plt.savefig(f'figures/ppc_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()

    if trace_plot:
        az.plot_trace(trace, var_names=["tau"])
        plt.tight_layout()
        plt.savefig(f'figures/traceplot_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()

    if convergence_check:
        summary = az.summary(trace, var_names=["tau"])
        print(summary)

    expanded_trace = trace.copy()
    expanded_trace.posterior = trace.posterior.expand_dims(pred_id=n_samples_per_trace)
    n_obs = len(a)

    with model:
        pm.set_data({"a": np.array([q98_a] * n_obs)})
        posterior_pred_q98 = pm.sample_posterior_predictive(expanded_trace, var_names=["y_obs"],
                                                            sample_dims=["chain", "draw", "pred_id"],
                                                            predictions=True, random_seed=42)
        pm.set_data({"a": np.array([q0_a] * n_obs)})
        posterior_pred_q0 = pm.sample_posterior_predictive(expanded_trace, var_names=["y_obs"],
                                                           sample_dims=["chain", "draw", "pred_id"],
                                                           predictions=True, random_seed=42)

    collapsed_posterior_pred_q98 = np.array([
        pt.inverse_transform(entry.reshape(-1, 1)).flatten()
        for entry in posterior_pred_q98.predictions['y_obs'].mean(axis=2).values.reshape(-1, n_obs)
    ])

    collapsed_posterior_pred_q0 = np.array([
        pt.inverse_transform(entry.reshape(-1, 1)).flatten()
        for entry in posterior_pred_q0.predictions['y_obs'].mean(axis=2).values.reshape(-1, n_obs)
    ])
    # Free memory
    del posterior_pred_q98
    del posterior_pred_q0

    dirichlet_weights = np.random.dirichlet(alpha=np.ones(n_obs), size=n_BB)
    ate_sample = np.matmul(dirichlet_weights, (collapsed_posterior_pred_q98 - collapsed_posterior_pred_q0).transpose()).flatten()
    del collapsed_posterior_pred_q98, collapsed_posterior_pred_q0

    ate_samples_np = np.random.choice(ate_sample, size=n_post, replace=False)
    del ate_sample

    return ate_samples_np


# parametric model
def causal_lm(X, a, y, q98_a, q0_a, cause_gene ="FOXP3",target_gene = "NKG7", n_samples=1000, plot_diagonostics=False, weights = None):
    X_design = np.hstack([np.ones((len(a), 1)), a[:, np.newaxis], X])
    X_design_wo_intcpt = X_design[:, 1:]

    model = LinearRegression().fit(X_design_wo_intcpt, y)
    delta_hat, gamma_hat = model.coef_[0], model.coef_[1:]

    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    V_hat = XtX_inv

    a_diff = q98_a - q0_a
    ATE_posterior_mean = delta_hat * a_diff
    ATE_posterior_variance = V_hat[1, 1] * (a_diff ** 2)
    ATE_posterior_std = np.sqrt(ATE_posterior_variance)

    fitted_values = model.predict(X_design_wo_intcpt)
    residuals = y - fitted_values

    model_stats = OLS(y, add_constant(X_design_wo_intcpt)).fit()

    if plot_diagonostics:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Residuals')

        plt.subplot(1, 2, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('QQ-Plot of Residuals')

        plt.tight_layout()
        plt.savefig(f'figures/lm_diagnostic_{cause_gene}_{target_gene}.pdf', dpi=300)
        plt.show()

        print(f"Posterior mean of ATE: {ATE_posterior_mean}")
        print(f"Posterior standard deviation of ATE: {ATE_posterior_std}")
        print(f"95% credible interval for ATE: [{ATE_posterior_mean - 1.96 * ATE_posterior_std}, {ATE_posterior_mean + 1.96 * ATE_posterior_std}]")
        print(f"R-squared: {model_stats.rsquared}")
        print(f"Adjusted R-squared: {model_stats.rsquared_adj}")

    ate_samples_pm = np.random.normal(loc=ATE_posterior_mean, scale=ATE_posterior_std, size=n_samples)

    return ate_samples_pm
def subsample_ate(a, testX_hat, testX, y, summary, n_post, start_size=50, n_SS=1):
    post_summary_lm = {}
    post_summary_bart = {}
    post_summary_gnpp = {}
    summary_gbf = {}
    n_obs = len(a)
    sizes = list(range(start_size, n_obs, 50)) + [n_obs]

    for size in sizes:
        all_summary_lm = np.empty(n_SS, dtype=float)
        all_summary_bart = np.empty(n_SS, dtype=float)
        all_summary_gnpp = np.empty(n_SS, dtype=float)
        gbFs = np.empty(n_SS, dtype=float)
        for i in range(n_SS):
            idx = np.random.choice(np.arange(n_obs), size=size, replace=False)

            subsample_testX_hat = testX_hat[idx, :]
            subsample_testX = testX[idx, :]
            subsample_y = y[idx]
            subsample_a = a[idx]

            q98_a = np.percentile(subsample_a, 98)
            q0_a = np.percentile(subsample_a, 0)

            ate_samples_lm = causal_lm(subsample_testX_hat, subsample_a, subsample_y, q98_a, q0_a, "TCF7", n_post)
            all_summary_lm[i] = summary(ate_samples_lm)
            del ate_samples_lm

            ate_samples_bart = causal_BART(subsample_testX, subsample_a, subsample_y, q98_a, q0_a, n_post=n_post,
                                           n_samples_per_trace=30, n_BB=1000, trace_plot=False)
            all_summary_bart[i] = summary(ate_samples_bart)

            gbF = compute_gBayesFactor(subsample_testX_hat, subsample_a, subsample_y, transform=True, n_subsample = 200)
            all_summary_gnpp[i] = gbF * all_summary_lm[i] + (1 - gbF) * all_summary_bart[i]
            gbFs[i] = gbF
            del subsample_testX_hat, subsample_testX, subsample_a, subsample_y, ate_samples_bart

        post_summary_lm[size] = all_summary_lm
        post_summary_bart[size] = all_summary_bart
        post_summary_gnpp[size] = all_summary_gnpp
        summary_gbf[size] = gbFs

    return post_summary_lm, post_summary_bart, post_summary_gnpp, summary_gbf