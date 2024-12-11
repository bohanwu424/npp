import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import subprocess
import datetime
import os

import npp.PolyaTree as pt

sample_tests = False
# Outputs.
outf = 'test_plotTree'
if not os.path.exists(outf):
    os.makedirs(outf)
batchTime = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
np.random.seed(10)


def test_plotTree():

    Finv = stats.norm.ppf
    n = 10
    X = sorted(stats.norm.rvs(size=n))

    root, X_eps_mstar = pt.build_btree(X, Finv)

    pt.plot_tree(root, X_eps_mstar, X)

    plt.savefig('{}/test_plotTree_{}.png'.format(outf, batchTime))


def test_marginal():

    h = 2.3

    FtildeInv = lambda u: np.sqrt(u)
    f = lambda x: 1
    fln = lambda x: 0
    F = lambda x: x
    Finv = lambda u: u

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

    tst_marg = pt.PolyaTree(X, FtildeInv).marginalln(fln, F, h)

    assert np.allclose(chk_marg, tst_marg)

    h = 2.3

    FtildeInv = lambda u: np.sqrt(u)
    f = lambda x: 3*np.power(x, 2)
    fln = lambda x: np.log(3) + 2*np.log(x)
    F = lambda x: np.power(x, 3)
    Finv = lambda u: np.power(u, 1/3)

    x_sp_null = FtildeInv(1/2)
    x_sp_0 = FtildeInv(1/4)
    x_sp_00 = FtildeInv(1/8)
    x_sp_001 = FtildeInv(3/16)

    muB_0 = F(x_sp_null) - F(0.)
    muB_1 = F(1.) - F(x_sp_null)
    muB_00 = F(x_sp_0) - F(0.)
    muB_01 = F(x_sp_null) - F(x_sp_0)
    muB_000 = F(x_sp_00) - F(0.)
    muB_001 = F(x_sp_0) - F(x_sp_00)
    muB_0010 = F(x_sp_001) - F(x_sp_00)
    muB_0011 = F(x_sp_0) - F(x_sp_001)
    d_null = (1/h)*np.power(f(x_sp_null)/1.0, 2)
    d_0 = (1/h)*np.power(f(x_sp_0)/muB_0, 2)
    d_00 = (1/h)*np.power(f(x_sp_00)/muB_00, 2)
    d_001 = (1/h)*np.power(f(x_sp_001)/muB_001, 2)
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
    X = [stats.uniform.rvs(loc=0, scale=x_sp_00),
         stats.uniform.rvs(loc=x_sp_00, scale=x_sp_001 - x_sp_00),
         stats.uniform.rvs(loc=x_sp_001, scale=x_sp_0 - x_sp_001)]

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

    mod = pt.PolyaTree(X, FtildeInv)
    pt.plot_tree(mod.broot, mod.X_eps_mstar, X)
    # plt.savefig('{}/test_marg_tree_{}.png'.format(outf, batchTime))

    tst_marg = pt.PolyaTree(X, FtildeInv).marginalln(fln, F, h)

    assert np.allclose(chk_marg, tst_marg)


def test_postpredictln():

    # This is a poor test (essentially the same as the code), given how
    # it is currently written; but if speedup is used, it will be actually
    # orthogonal.

    FtildeInv = lambda u: np.sqrt(u)
    f = lambda x: 3*np.power(x, 2)
    fln = lambda x: np.log(3) + 2*np.log(x)
    F = lambda x: np.power(x, 3)
    Finv = lambda u: np.power(u, 1/3)

    h = 1/3
    n = 10
    X = sorted(stats.uniform.rvs(size=n))
    xnew = stats.uniform.rvs()
    print(X, xnew)
    margX = pt.PolyaTree(X, FtildeInv).marginalln(fln, F, h)
    prednew = pt.PolyaTree(X, FtildeInv).postpredictln(xnew, fln, F, h)
    print(margX, prednew)
    tst_pred = margX + prednew

    Xnew = np.array(sorted(list(X) + [xnew]))
    chk_pred = pt.PolyaTree(Xnew, FtildeInv).marginalln(fln, F, h)

    assert np.allclose(tst_pred, chk_pred)


def test_plotPredictive():

    FtildeInv = lambda u: np.sqrt(u)
    f = lambda x: 3*np.power(x, 2)
    fln = lambda x: np.log(3) + 2*np.log(x)
    F = lambda x: np.power(x, 3)

    hs = [0.1, 1., 10.]
    n = 10
    X = np.sort(stats.uniform.rvs(size=n))
    xnews = np.linspace(0, 1, 100)[1:-1]

    mod = pt.PolyaTree(X, FtildeInv)
    pt.plot_tree(mod.broot, mod.X_eps_mstar, X)
    plt.savefig('{}/test_postpredict_tree_{}.png'.format(outf, batchTime))

    for h in hs:
        pt.plotPredictive(xnews, X, FtildeInv, fln, F, h)
        plt.savefig('{}/test_postpredict_plot_h{}_{}.png'.format(
                        outf, h, batchTime))