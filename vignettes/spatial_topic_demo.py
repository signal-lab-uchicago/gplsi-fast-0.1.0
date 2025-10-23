
# Spatial topic demo: graph-aware GpLSI vs plain pLSI
# Usage:
#   1) pip install ./gplsi_fast_pkg   (or use your installed gplsi-fast)
#   2) python spatial_topic_demo.py

import sys, types, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp, scipy.sparse.linalg as spla
from numpy.linalg import norm

# Optional fallback if pycvxcluster isn't installed: light Laplacian smoother
try:
    import pycvxcluster.pycvxcluster as _pc
except Exception:
    pkg = types.ModuleType("pycvxcluster"); sub = types.ModuleType("pycvxcluster.pycvxcluster")
    class SSNAL:
        def __init__(self, gamma=1.0, **kw): self.gamma=gamma; self.kwargs={}
        def fit(self, X, weight_matrix=None, **kw):
            n = X.shape[0]
            W = sp.csr_matrix(weight_matrix) if weight_matrix is not None else sp.eye(n, format="csr")
            d = np.asarray(W.sum(axis=1)).ravel(); L = sp.diags(d) - W
            A = sp.eye(n, format="csr") + self.gamma * L
            solve = spla.factorized(A.tocsc())
            X_smooth = np.column_stack([solve(X[:, j]) for j in range(X.shape[1])])
            self.centers_ = X_smooth.T; self.y_=None; self.z_=None; return self
    sub.SSNAL = SSNAL
    sys.modules["pycvxcluster"] = pkg; sys.modules["pycvxcluster.pycvxcluster"] = sub
    sys.modules["pycvxcluster"].pycvxcluster = sub

from gplsi_fast import GpLSI_
from scipy.optimize import linear_sum_assignment

def make_spatial_topics(n_side=18, K=3, V=80, seed=11):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0,1,n_side), np.linspace(0,1,n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]); n = coords.shape[0]
    centers = np.array([[0.2,0.2],[0.8,0.3],[0.5,0.8]])[:K]
    A_true = np.zeros((K, V))
    for k in range(K):
        base = rng.dirichlet(0.4*np.ones(V)); heavy = rng.choice(V, size=V//8, replace=False)
        base[heavy] += 2.0/V; A_true[k] = base/base.sum()
    tau = 0.14
    D2 = np.stack([np.sum((coords-c)**2, axis=1) for c in centers], axis=1)
    logits = -D2/(2*tau*tau); logits -= logits.max(axis=1, keepdims=True)
    W_true = np.exp(logits); W_true /= W_true.sum(axis=1, keepdims=True)
    N_per = 100; P = W_true @ A_true
    X = np.stack([rng.multinomial(N_per, P[i]) for i in range(n)], axis=0).astype(float)
    return sp.csr_matrix(X), W_true, A_true, coords

def knn_graph(coords, k=6):
    n = coords.shape[0]
    D2 = np.sum(coords**2, axis=1, keepdims=True) + np.sum(coords**2, axis=1) - 2*coords@coords.T
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1)[:, :k]
    med = np.median(np.sqrt(D2[np.arange(n)[:,None], idx])); sigma = med if med>0 else 1e-3
    rows=[]; cols=[]; ws=[]
    for i in range(n):
        for j in idx[i]:
            w = float(np.exp(-D2[i,j]/(2*sigma*sigma)))
            rows += [i,j]; cols += [j,i]; ws += [w,w]
    W = sp.coo_matrix((ws,(rows,cols)), shape=(n,n)).tocsr(); W = W.maximum(W.T)
    edges = np.vstack(W.nonzero()).T
    edge_df = pd.DataFrame({"src": edges[:,0], "tgt": edges[:,1]})
    return W, edge_df

def align_and_metrics(W_true, A_true, W_hat, A_hat, W):
    def row_cos(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = row_cos(A_hat) @ row_cos(A_true).T
    r, c = linear_sum_assignment(1 - np.abs(S))
    P = np.zeros_like(S); P[r, c] = 1
    A_hat_aligned = P.T @ A_hat
    W_hat_aligned = W_hat @ P
    y_true = np.argmax(W_true, axis=1); y_hat = np.argmax(W_hat_aligned, axis=1)
    acc = (y_true == y_hat).mean()
    fro_err = np.linalg.norm(W_true - W_hat_aligned) / (np.linalg.norm(W_true) + 1e-12)
    A_true_norm = row_cos(A_true); A_hat_norm = row_cos(A_hat_aligned)
    topic_sim = np.diag(A_hat_norm @ A_true_norm.T).mean()
    i, j = W.nonzero(); w = np.asarray(W[i,j]).ravel(); keep = i < j
    edge_disagree = np.average((y_hat[i] != y_hat[j])[keep], weights=w[keep])
    return acc, fro_err, topic_sim, edge_disagree, y_hat

def scatter(ax, coords, labels, title):
    ax.scatter(coords[:,0], coords[:,1], s=9, c=labels)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

if __name__ == "__main__":
    X, W_true, A_true, coords = make_spatial_topics()
    W, edge_df = knn_graph(coords)

    graph = GpLSI_(method="two-step", precondition="whiten", verbose=0, maxiter=8).fit(X, X.shape[0], 3, edge_df, W)
    plsi  = GpLSI_(method="pLSI",    precondition="whiten", verbose=0).fit(X, X.shape[0], 3, edge_df, W)

    acc_p, fro_p, sim_p, ed_p, y_plsi  = align_and_metrics(W_true, A_true, plsi.W_hat,  plsi.A_hat,  W)
    acc_g, fro_g, sim_g, ed_g, y_graph = align_and_metrics(W_true, A_true, graph.W_hat, graph.A_hat, W)

    print(f"pLSI:   acc={acc_p:.3f} topic_cosine={sim_p:.3f} fro_err_W={fro_p:.3f} edge_disagree={ed_p:.3f}")
    print(f"Graph:  acc={acc_g:.3f} topic_cosine={sim_g:.3f} fro_err_W={fro_g:.3f} edge_disagree={ed_g:.3f}")

    import matplotlib.pyplot as plt
    y_true = np.argmax(W_true, axis=1)

    plt.figure(figsize=(4,4)); scatter(plt.gca(), coords, y_true, "Ground truth topics"); plt.tight_layout(); plt.savefig("demo_true.png")
    plt.figure(figsize=(4,4)); scatter(plt.gca(), coords, y_plsi, "pLSI inferred topics"); plt.tight_layout(); plt.savefig("demo_plsi.png")
    plt.figure(figsize=(4,4)); scatter(plt.gca(), coords, y_graph, "Graph-aware GpLSI inferred topics"); plt.tight_layout(); plt.savefig("demo_graph.png")
    print("Saved demo_true.png, demo_plsi.png, demo_graph.png")
