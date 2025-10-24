
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds

from .graphsvd import graphSVD

try:
    import cvxpy as cp  # optional
except Exception:  # pragma: no cover
    cp = None


class GpLSI_(object):
    """
    GpLSI with improved preconditioning and plumbing.

    Parameters
    ----------
    lambd : float or None
    lamb_start : float
    step_size : float
    grid_len : int
    maxiter : int
    eps : float
    method : {"two-step", "pLSI"}
    return_anchor_docs : bool
    verbose : int
    precondition : {"none", "whiten", "cvx"} or bool
        - "whiten" (default) uses closed-form Q = (M M^T)^(-1/2)
        - "none" disables preconditioning
        - "cvx" uses the original log-det cone program (requires cvxpy)
    initialize : bool
    """
    def __init__(
        self,
        lambd=None,
        lamb_start=1e-4,
        step_size=1.2,
        grid_len=29,
        maxiter=50,
        eps=1e-5,
        method="two-step",
        return_anchor_docs=True,
        verbose=0,
        precondition="none",
        initialize=True,
    ):
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.maxiter = maxiter
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        # Back-compat for boolean precondition flag
        if isinstance(precondition, bool):
            precondition = "whiten" if precondition else "none"
        self.precondition = precondition
        self.initialize = initialize

    # ----------------------- public API -----------------------
    def fit(self, X, N, K, edge_df, weights):
        if self.method == "pLSI":
            if self.verbose:
                print("Running pLSI...")
            U, s, VT = svds(X, k=K)
            self.U, self.L, self.V = U, s, VT.T
            self.U_init = None
        else:
            if self.verbose:
                print("Running graph aligned pLSI...")
            (
                self.U,
                self.V,
                self.L,
                self.U_init,
                self.V_init,
                self.L_init,
                self.lambd,
                self.lambd_errs,
                self.used_iters,
            ) = graphSVD(
                X,
                N,
                K,
                edge_df,
                weights,
                self.lamb_start,
                self.step_size,
                self.grid_len,
                self.maxiter,
                self.eps,
                self.verbose,
                self.initialize,
            )

        if self.verbose:
            print("Running SPOC...")
        J, H_hat = self.preconditioned_spa(self.U, K, self.precondition)

        self.W_hat = self.get_W_hat(self.U, H_hat)
        self.A_hat = self.get_A_hat(self.W_hat, X)
        if self.return_anchor_docs:
            self.anchor_indices = J

        if self.U_init is not None:
            J_init, H_hat_init = self.preconditioned_spa(self.U_init, K, self.precondition)
            self.W_hat_init = self.get_W_hat(self.U_init, H_hat_init)
            self.A_hat_init = self.get_A_hat(self.W_hat_init, X)

        return self

    # ----------------------- SPOC helpers -----------------------
    @staticmethod
    def preprocess_U(U, K):
        # Flip signs so first element is non-negative for each column
        U2 = U.copy()
        for k in range(K):
            if U2[0, k] < 0:
                U2[:, k] = -U2[:, k]
        return U2

    @staticmethod
    def _precondition_whiten(M):
        # Q = (M M^T)^(-1/2) via eigen
        S = M @ M.T  # K x K
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-12, None)
        Q = V @ (np.diag(1.0 / np.sqrt(w))) @ V.T
        return Q

    @staticmethod
    def _precondition_cvx(M):
        if cp is None:
            raise ImportError("cvxpy is not available. Install with extras: pip install 'gplsi-fast[cvx]'")
        K = M.shape[0]
        Q = cp.Variable((K, K), symmetric=True)
        objective = cp.Maximize(cp.log_det(Q))
        constraints = [cp.norm(Q @ M, axis=0) <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        return Q.value

    def preconditioned_spa(self, U, K, precondition="whiten"):
        J = []
        M = self.preprocess_U(U, K).T  # K x n
        if precondition == "whiten":
            L = self._precondition_whiten(M)
            S = L @ M
        elif precondition == "cvx":
            L = self._precondition_cvx(M)
            S = L @ M
        else:
            S = M

        # SPA selection
        for _ in range(K):
            norms = np.linalg.norm(S, axis=0)
            j = int(np.argmax(norms))
            s = S[:, [j]]  # Kx1
            # rank-1 projection to deflate
            S = S - (s @ (s.T @ S)) / (np.linalg.norm(s)**2 + 1e-12)
            J.append(j)

        H_hat = U[J, :]
        return J, H_hat

    # ----------------------- projections -----------------------
    def get_W_hat(self, U, H):
        projector = H.T @ np.linalg.inv(H @ H.T)
        theta = U @ projector
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    def get_A_hat(self, W_hat, M):
        projector = np.linalg.inv(W_hat.T @ W_hat) @ W_hat.T
        theta = projector @ M
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    @staticmethod
    def _euclidean_proj_simplex(v, s=1):
        n = v.shape[0]
        if v.sum() == s and np.all(v >= 0):
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        return w
