# Adaptive Nonparametric Perturbations of Parametric Bayesian Models
This repository contains code used in the paper

>Adaptive Nonparametric Perturbations of Parametric Bayesian Models

>Bohan Wu*, Eli N. Weinstein*, Sohrab Salehi, Yixin Wang, David M. Blei

>2024

# Installation

To install the required packages, you can run

`pip install .`

# Synthetic experiments

Code for the synthetic data experiments can be found in the folder `npp`. Bash scripts for the experiments on synthetic data can be found in the folder `npp/experiments/`.
In particular, the file `gnpp_demo_full.sh` runs the gNPP experiments, and the file `npp_demo_full.sh` runs the NPP experiments. The summmary plots were created with the notebook `summarize/summarize_demo.ipynb`.

# Causal effects of gene expression
Code for the real data experiments is collected in the `application` folder. The dataset can be found in the `data/` folder. The main files for running the experiments are named `gene_demo{num}.py` and are located in the `application` folder. The plots are generated using the notebook `application/summarize_app.ipynb`.