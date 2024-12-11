import numpy as np
from scipy import stats

import npp.PolyaTreeEmbedded as pt

np.random.seed(11)


def test_marginal():

    h = 2.3

    FtildeInv = lambda u: np.sqrt(u)
    f = lambda x: np.ones_like(x)

    d_null = 1/h
    d_0 = 2/h
    d_00 = 4/h
    d_001 = (1/h)*(8/np.power(np.sqrt(2) - 1, 2))
    muB_0 = 1/np.sqrt(2)
    muB_1 = (np.sqrt(2) - 1)/np.sqrt(2)
    muB_00 = 1/2
    muB_01 = 1/np.sqrt(2) - 1/2
    muB_000 = 1/(2*np.sqrt(2))
    muB_001 = 1/2 - 1/(2*np.sqrt(2))
    muB_0010 = np.sqrt(3)/4 - 1/(2*np.sqrt(2))
    muB_0011 = 1/2 - np.sqrt(3)/4
    c_null = muB_1/muB_0
    c_0 = muB_01/muB_00
    c_00 = muB_001/muB_000
    c_001 = muB_0011/muB_0010

    al_0 = d_null/np.sqrt(c_null)
    al_1 = d_null*np.sqrt(c_null)
    al_00 = d_0/np.sqrt(c_0)
    al_01 = d_0*np.sqrt(c_0)
    al_000 = d_00/np.sqrt(c_00)
    al_001 = d_00*np.sqrt(c_00)
    al_0010 = d_001/np.sqrt(c_001)
    al_0011 = d_001*np.sqrt(c_001)

    X = [(1/(2*np.sqrt(2)))/3,
         1/(2*np.sqrt(2)) + muB_0010*0.3,
         np.sqrt(3)/4 + muB_0011*0.7]

    term0 = f(X[0])
    term1 = (f(X[1]) *
             ((al_0 + 1)*(al_0 + al_1)/(al_0*(al_0 + al_1 + 1))) *
             ((al_00 + 1)*(al_00 + al_01)/(al_00*(al_00 + al_01 + 1))) *
             ((al_000 + al_001)/(al_000 + 1 + al_001)))
    term2 = (f(X[2]) *
             ((al_0 + 2)*(al_0 + al_1)/(al_0*(al_0 + al_1 + 2))) *
             ((al_00 + 2)*(al_00 + al_01)/(al_00*(al_00 + al_01 + 2))) *
             ((al_001 + 1)*(al_000 + al_001)/(al_001*(al_000 + al_001 + 2))) *
             ((al_0010 + al_0011)/(al_0010 + al_0011 + 1)))

    chk_marg = np.log(term0 + 1e-32) + np.log(term1) + np.log(term2)

    # Set up embedded PT model.
    param_prior = stats.norm
    def Uniform(dummy_theta):
        return stats.uniform(loc=np.zeros_like(dummy_theta))

    class Square:
        def __init__(self):
            self.ppf = FtildeInv

    class Delta:
        def __init__(self, val):
            self.val = val
        def rvs(self, size=1):
            return np.ones(size) * self.val

    param_like = Uniform
    partition_distribution = Square()
    importance_distribution = param_prior
    h_distribution = Delta(h)
    support = (0., 1.)
    nsamples = 2

    pte = pt.PolyaTreeEmbedded(param_prior, param_like, partition_distribution,
                               h_distribution, nsamples, support)
    bf_ln = pte.bayesfactor_ln(np.array(X), importance_distribution)
    tst_marg = -bf_ln  # Numerator is ignored, as parametric model is fixed at uniform.

    assert np.allclose(chk_marg, tst_marg)


def test_postpredictln():
    # Set up embedded PT model.
    FtildeInv = lambda u: np.sqrt(u)
    param_prior = stats.norm

    def Uniform(dummy_theta):
        return stats.uniform(loc=np.zeros_like(dummy_theta))

    class Square:
        def __init__(self):
            self.ppf = FtildeInv

    class Delta:
        def __init__(self, val):
            self.val = val

        def rvs(self, size=1):
            return np.ones(size) * self.val

    param_like = Uniform
    partition_distribution = Square()
    importance_distribution = param_prior
    h = 1.3
    h_distribution = Delta(h)
    support = (0., 1.)
    nsamples = 2

    # Data.
    X = stats.uniform.rvs(size=4)
    nnew = 10
    Xnew = stats.uniform.rvs(size=nnew)

    pte = pt.PolyaTreeEmbedded(param_prior, param_like, partition_distribution,
                               h_distribution, nsamples, support)
    base_marg = -pte.bayesfactor_ln(np.array(X), importance_distribution)
    post_predict_ln_tst = pte.posterior_predictive_ln(Xnew)

    post_predict_ln_chk = np.zeros(nnew) - base_marg
    for j in range(nnew):
        pte = pt.PolyaTreeEmbedded(param_prior, param_like, partition_distribution,
                                   h_distribution, nsamples, support)
        post_predict_ln_chk[j] += -pte.bayesfactor_ln(np.array(list(X) + [Xnew[j]]), importance_distribution)
    assert np.allclose(post_predict_ln_tst, post_predict_ln_chk)