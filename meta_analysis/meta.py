import numpy as np
import pandas as pd

class MetaAnalysis:

    def generate_meta_estimate(self, df, balanced_matrix):
        # Precision weighted estimates
        precision = df["lfcSE"].pow(-2)

        # importance weights, (arbitrary) 
        importance_weights = (balanced_matrix.T * df["baseMean"]).T

        # final weights (importances and precision)
        weights = importance_weights * precision.values[:, None]

        w = weights.sum(axis=0)

        # Normalized parameter estimate, unshrunk
        normalized_estimate = (weights * df["log2FoldChange"].values[:, None]).sum(axis=0) / w

        # Normalized standard error, unshrunk
        normalized_standard_error = np.sqrt(1 / w)

        # DerSimonian-Laird for correction of between term variance.
        tau2 = self.dersimonian_laird_tau2(normalized_estimate, normalized_standard_error, weights=weights.sum(axis=0))

        # Weighted estimate of the mean for the entire dataset
        full_dataset_weights = 1 / (normalized_standard_error.pow(2) + tau2)
        full_dataset_mean = (normalized_estimate * full_dataset_weights).sum()
        full_dataset_mean /= full_dataset_weights.sum()

        full_dataset_variance = (1 / np.sqrt(full_dataset_weights.sum()))

        # Estimate Shrinkage
        v = normalized_standard_error**2

        # shrinkage factor
        B = tau2 / (tau2 + v)

        # shrinked means
        shrunken_estimate = B * normalized_estimate #+ (1 - B) * full_dataset_mean

        # shrinked variance
        shrunken_variance = (tau2 * v) / (tau2 + v)
        shrunken_se = np.sqrt(shrunken_variance)

        return {
            "weights": weights.sum(axis=0),
            "normalized_estimate": normalized_estimate,
            "normalized_standard_error": normalized_standard_error,
            "shrunken_estimate": shrunken_estimate,
            "shrunken_se": shrunken_se,
        }

    @staticmethod
    def dersimonian_laird_tau2(x, var, weights=None):
        """
        DerSimonian-Laird estimator of tau^2 (random effects variance).

        Parameters
        ----------
        x : array-like
            Effect sizes.

        var : array-like
            Within-study variances (SE^2).

        weights : array-like or None
            Optional external weights.
            If None, uses inverse-variance weights (1/var).

        Returns
        -------
        tau2 : float
            Estimated between-study variance.
        """

        x = np.asarray(x)
        var = np.asarray(var)

        if weights is None:
            w = 1.0 / var
        else:
            w = np.asarray(weights)

        W = w.sum()
        W2 = (w**2).sum()

        # fixed-effect mean under weights
        theta = (w * x).sum() / W

        # Cochran's Q
        Q = (w * (x - theta)**2).sum()

        # correction term
        denom = W - (W2 / W)

        tau2 = (Q - (len(x) - 1)) / denom

        return max(tau2, 0.0)