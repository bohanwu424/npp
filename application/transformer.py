# The transformer function F(X) -> Z

import argparse
import numpy as np
import scipy
import pickle


def transform_counts(X, pca_obj=None, standardizer_obj=None, libsize=10000):
    """Givne counts, return the embedding."""
    # Get sizes
    D = X.shape[1]
    pca_input_shape = pca_obj.components_.shape[1]
    assert D == pca_input_shape, f"X.shape[1] = {D} != pca_obj.components_.shape[1] = {pca_input_shape}"  
    # Library size normalization.
    X = (X / X.sum(axis=1)[:, None]) * libsize
    # Log transformation.
    X = np.log1p(X)
    # Standardize.
    X_norm = standardizer_obj.transform(X)
    # PCA.
    X_hat = pca_obj.transform(X_norm)
    return X_hat


def test_run(cause_gene = 'TCF7', target_gene = 'SELL'):
    """
    Given a reference, and test dataset:

    1. Load
    2. Separate out the cause, target, and confounding genes
    3. Return a, y, and f(X), and X
    """
    ref_genes_path = '../data/dominguez/processed/items/genes.txt'
    test_path = '../data/vazquez/processed/items/counts.npz'
    test_genes_path = '../data/vazquez/processed/items/genes.txt'
    test_cells_path = '../data/vazquez/processed/items/cells.txt'
    pca_path = '../data/dominguez/processed/data/pca.pkl'
    standardizer_path = '../data/dominguez/processed/data/scaler.pkl'

    # Load
    ref_genes = np.loadtxt(ref_genes_path, dtype=str)
    test = scipy.sparse.load_npz(test_path).toarray()  # .A
    test_genes = np.loadtxt(test_genes_path, dtype=str)
    test_cells = np.loadtxt(test_cells_path, dtype=str)
    
    with open(pca_path, 'rb') as f:
        pca_obj = pickle.load(f)
    with open(standardizer_path, 'rb') as f:
        standardizer_obj = pickle.load(f)
    
    # Separate out the cause, target, and confounding genes.
    confounding_genes = ref_genes.tolist()

    # Find the index of confounding genes in the test dataset.
    indx = []
    for gene in confounding_genes:
        tmp_indx = np.where(test_genes == gene)[0][0]
        indx.append(tmp_indx)

    testX = test[:, indx]
    testX_hat = transform_counts(testX, pca_obj, standardizer_obj)

    # Return cause and target.
    a = test[:, test_genes == cause_gene].flatten()
    y = test[:, test_genes == target_gene].flatten()

    return a, y, testX_hat, testX
