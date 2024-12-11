import numpy as np
import matplotlib.pyplot as plt

# Part 1: Creating the N tree. Initially, add first point to stack.
# Then, for next point x_1, consider each decision for eps(x_1) in sequence.
# Assign each point to corresponding N_{eps, 0}, N_{eps, 1}. If in the same
# direction, continue. Otherwise stop.
# Now for point x_i. Consider each decision for eps(x_i) in sequence.
# Add to corresponding N until a) a decision point is reached with positive
# N_k in one direction (k in 0, 1) and zero in the direction that the new
# point goes, or b) both Ns are zero. In case (a), just add to the new
# N_k and done. In case (b), use previous subroutine, extending the tree
# until the two points go different ways.
# Since our implementation is one-dimensional, useful to sort points initially,
# so now you know that the nearest point (eligible for b subroutine) is the
# previous.


class B_node:

    def __init__(self, k=0, m=0, iv=(-np.inf, np.inf), split=0., counts=0):

        self.k = k
        self.m = m
        self.iv = iv
        self.split = split
        self.counts = counts
        self.kids = [None, None]

    def indx(self, eps):

        if eps == []:
            return self

        else:
            ep = eps[0]
            return self.kids[ep].indx(eps[1:])

    def fill_kids(self, Finv, n):

        lims = [self.iv[0], self.split, self.iv[1]]
        splitL = Finv((4*self.k + 1)/np.power(2., self.m+2))
        splitR = Finv((4*self.k + 3)/np.power(2., self.m+2))

        self.kids = [
                B_node(2*self.k, self.m+1, lims[:2], splitL, np.zeros(n+1)),
                B_node(2*self.k+1, self.m+1, lims[1:], splitR, np.zeros(n+1))]

    def get_kid(self, x):

        return int(x > self.kids[0].iv[1])

    def get_root(self, Finv, n):

        self.k = 0
        self.m = 0
        self.iv = ((Finv(0.0), Finv(1.0)))
        self.split = Finv(1/2)
        # n+1 instead of n for posterior predictive calculations
        self.counts = np.zeros(n+1)
        self.kids = [None, None]

        return self

    def add_count(self, j):

        self.counts[j:] = self.counts[j:] + 1


def add_x_btree(x, j, prev_x, bnode, Finv, n, eps=[]):

    if bnode.kids[0] is None:
        bnode.fill_kids(Finv, n)

        prev_x_kid = bnode.get_kid(prev_x)
        bnode.kids[prev_x_kid].add_count(j-1)

    x_kid = bnode.get_kid(x)
    bnode.kids[x_kid].add_count(j)
    eps.append(x_kid)
    if bnode.kids[x_kid].counts[j] == 1:
        return eps
    else:
        return add_x_btree(x, j, prev_x, bnode.kids[x_kid], Finv, n, eps)


def build_btree(X, Finv):

    X = sorted(X)
    n = len(X)
    prev_x = X[0]
    root = B_node().get_root(Finv, n)
    root.add_count(0)
    X_eps_mstar = [[]]
    for i in range(1, n):
        X_eps_mstar.append(add_x_btree(X[i], i, prev_x, root, Finv, n, []))
        prev_x = X[i]
        root.add_count(i)

    return root, X_eps_mstar


# ----
# These three functions obtain parameters for the beta distributions,
# based on the distribution pdf f(x|theta), its cdf F, and the hyperparameter
# h, along with the previously constructed tree F_tilde.

def log_d_eps(bnode, fln, F, h):

    return -np.log(h) + 2*(fln(bnode.split)
                           - np.log(F(bnode.iv[1]) - F(bnode.iv[0])))


def log_c_eps(bnode, F):

    return (np.log(F(bnode.iv[1]) - F(bnode.split))
            - np.log(F(bnode.split) - F(bnode.iv[0])))


def log_alphas_eps(eps, broot, fln, F, h):

    bnode = broot.indx(eps[:-1])
    log_d = log_d_eps(bnode, fln, F, h)
    log_c = log_c_eps(bnode, F)
    return [log_d - 0.5*log_c, log_d + 0.5*log_c]


# ----
# Convenient shorthand
def lse2(a, b):

    return np.logaddexp(a, b)  # logsumexp(np.array([a, b]))


def lse2expb(a, expb):

    if expb == 0:
        return a
    else:
        return lse2(a, np.log(expb))


# ----
def marginal_xj(x, j, eps, broot, fln, F, h):
    logp = fln(x)
    for m in range(len(eps)):
        eps_m = eps[:(m+1)]
        lna = log_alphas_eps(eps_m, broot, fln, F, h)
        N = [kid.counts[j-1] for kid in broot.indx(eps_m[:-1]).kids]
        lnap = [lse2expb(lna[ii], N[ii]) for ii in range(2)]
        ep = eps_m[-1]

        term = (lnap[ep] + lse2(lna[0], lna[1])
                - lna[ep] - lse2(lnap[0], lnap[1]))
        logp = logp + term
    return logp


# Complete model.
class PolyaTree:

    def __init__(self, X, FtildeInv):
        # Input:
        # X: data (1D numpy array)
        # FtildeInv: the inverse cdf of the distribution used for setting the tree partitions.

        self.X = np.sort(X)
        self.FtildeInv = FtildeInv
        self.broot, self.X_eps_mstar = build_btree(X, FtildeInv)

    def marginalln(self, fln, F, h):
        # Input:
        # fln: log likelihood of base distribution.
        # F: cdf of base distribution.
        # h: inverse concentration parameter

        n = len(self.X)
        logp = fln(self.X[0])
        for j in range(1, n):
            eps = self.X_eps_mstar[j]
            term = marginal_xj(self.X[j], j, eps, self.broot, fln, F, h)
            logp = logp + term

        return logp

    def postpredictln(self, xnew, fln, F, h):

        # ** Unnecessarily slow. A better solution would be to modify
        # add_x_btree to accept upper and lower neighbors.

        marg0 = self.marginalln(fln, F, h)

        Xnew = np.array(sorted(list(self.X) + [xnew]))
        marg1 = PolyaTree(Xnew, self.FtildeInv).marginalln(fln, F, h)

        return marg1 - marg0


# ----
# Plotting tools

def plot_tree(root, X_eps_mstar, X, s_1=40):
    nodes, branches, finalInfo = node_branch_list(root)
    nodes_m = np.array([el[0] for el in nodes])
    nodes_loc = np.array([el[1] for el in nodes])
    nodes_counts = np.array([el[2] for el in nodes])

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(nodes_m, nodes_loc, s=s_1*nodes_counts[:, -1], c='k')
    for m, loc, counts in nodes:
        plt.text(m, loc, str(counts), horizontalalignment='center',
                 fontsize=12)

    for line_x, line_y in branches:
        plt.plot(line_x, line_y, 'k-', linewidth=1)

    m_lim = np.max(nodes_m) + 1
    plt.scatter(m_lim*np.ones(len(X)), X, s=s_1, c='r')
    for eps_mstar, x in zip(X_eps_mstar, X):
        plt.text(m_lim, x, str(eps_mstar), horizontalalignment='right',
                 verticalalignment='center', fontsize=13)

    for m, iv in finalInfo:

        plt.plot((m, m_lim), (iv[1], iv[1]), 'g--', linewidth=2)
        plt.plot((m, m_lim), (iv[0], iv[0]), 'b:', linewidth=2)

    plt.xlabel('m', fontsize=16)
    plt.xticks(np.arange(m_lim), fontsize=13)
    plt.ylabel('x', fontsize=16)
    plt.yticks(fontsize=13)
    plt.title('Interval Tree', fontsize=16)
    plt.tight_layout()

    return fig


def node_branch_list(bnode):

    nodeLoc = bnode.split
    nodeID = [(bnode.m, nodeLoc, bnode.counts)]

    if bnode.kids[0] is None:
        final_branch = [(bnode.m, bnode.iv)]
        return nodeID, [], final_branch

    else:
        nodeLocL = bnode.kids[0].split
        nodeLocR = bnode.kids[1].split
        branchIDs = [((bnode.m, bnode.kids[0].m), (nodeLoc, nodeLocL)),
                     ((bnode.m, bnode.kids[1].m), (nodeLoc, nodeLocR))]

        nodeL, branchL, finalL = node_branch_list(bnode.kids[0])
        nodeR, branchR, finalR = node_branch_list(bnode.kids[1])
        return (nodeID + nodeL + nodeR, branchIDs + branchL + branchR,
                finalL + finalR)


def plotPredictive(xnews, X, FtildeInv, fln, F, h):

    model = PolyaTree(X, FtildeInv)
    ys = np.zeros(len(xnews))
    for i, xnew in enumerate(xnews):
        ys[i] = model.postpredictln(xnew, fln, F, h)

    fig = plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(xnews, ys, 'bo')
    plt.plot(xnews, fln(xnews), 'k-', linewidth=1)
    plt.ylabel(r'$\log p(x|\{x_1, ..., x_n\})$')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title('Posterior Predictive, h = {}'.format(h), fontsize=16)
    plt.legend(['posterior predictive', 'base distribution'])
    xlims0 = plt.xlim()

    plt.subplot(2, 1, 2)
    plt.plot(X, np.zeros(len(X)), 'ro')
    plt.xlim(xlims0)
    plt.xlabel('x', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks([])
    plt.tight_layout()

    return fig
# -----