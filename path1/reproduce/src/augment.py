"""
Gaussian Mixture + Gaussian Copula data augmentation for power time-series.

Approach:
- Fit per-column 1D GaussianMixture (select components by BIC up to k_max).
- Convert observations to uniforms via empirical CDF.
- Fit Gaussian copula (correlation of normal scores).
- Sample uniforms from copula, then map back to each column via inverse CDF
  of the fitted GMM (numerical inversion on a grid).

Notes:
- Designed for continuous variables (loads, renewable, price).
- Time-order is not modeled; sampling is i.i.d. per row. For multi-day
  augmented sets, we sample rows independently and reindex datetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


@dataclass
class GMMFit:
    weights: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    grid_x: np.ndarray
    grid_cdf: np.ndarray

    def sample_ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF by interpolation on precomputed grid."""
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.interp(u, self.grid_cdf, self.grid_x)


def fit_gmm_1d(x: np.ndarray, k_max: int = 5, grid_size: int = 2000) -> GMMFit:
    x = x.reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    for k in range(1, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gmm.fit(x)
        bic = gmm.bic(x)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    gmm = best_gmm
    xs = np.linspace(np.min(x) - 3 * np.std(x), np.max(x) + 3 * np.std(x), grid_size)
    cdf = np.zeros_like(xs)
    for w, m, c in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.flatten()):
        cdf += w * norm.cdf(xs, loc=m, scale=np.sqrt(c))
    return GMMFit(
        weights=gmm.weights_.copy(),
        means=gmm.means_.flatten().copy(),
        covars=gmm.covariances_.flatten().copy(),
        grid_x=xs,
        grid_cdf=cdf,
    )


def empirical_cdf_uniform(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    return (r - 0.5) / len(r)


def fit_gaussian_copula(U: np.ndarray) -> np.ndarray:
    """Fit correlation matrix on normal scores."""
    Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    corr = np.corrcoef(Z, rowvar=False)
    return corr


def sample_gaussian_copula(corr: np.ndarray, n: int) -> np.ndarray:
    d = corr.shape[0]
    L = np.linalg.cholesky(corr + 1e-6 * np.eye(d))
    z = np.random.randn(n, d) @ L.T
    return norm.cdf(z)


class GMCDataAugmentor:
    def __init__(self, df: pd.DataFrame, target_cols: List[str], k_max: int = 5):
        self.df = df.copy()
        self.target_cols = target_cols
        self.k_max = k_max
        self.gmms: Dict[str, GMMFit] = {}
        self.corr: np.ndarray | None = None

    def fit(self):
        U_list = []
        for col in self.target_cols:
            gmm_fit = fit_gmm_1d(self.df[col].to_numpy(), k_max=self.k_max)
            self.gmms[col] = gmm_fit
            u = empirical_cdf_uniform(self.df[col].to_numpy())
            U_list.append(u)
        U = np.stack(U_list, axis=1)
        self.corr = fit_gaussian_copula(U)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        assert self.corr is not None, "Call fit() first."
        U = sample_gaussian_copula(self.corr, n)
        data = {}
        for i, col in enumerate(self.target_cols):
            gmm_fit = self.gmms[col]
            data[col] = gmm_fit.sample_ppf(U[:, i])
        return pd.DataFrame(data)


__all__ = ["GMCDataAugmentor"]

