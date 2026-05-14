import numpy as np
import pandas as pd
from scipy import stats 
from scipy.stats import gaussian_kde, norm

class MetaAnalysis:

    def generate_meta_estimate(self, df, balanced_matrix):
        """
        Random-effects meta-analysis with importance-weighted inverse-variance pooling.
        """

        # empirical lfc and se
        se = df["lfcSE"].values
        lfc = df["log2FoldChange"].values

        # precision = 1 / se^2
        precision = 1.0 / (se ** 2)  # shape: (n_studies,)

        # Normalized Importances (across entire dataset).
        importance_weights = balanced_matrix * df["baseMean"].values[:, None]

        # w = a / se^2
        weights = precision[:, None] * importance_weights

        # naive estimate
        normalized_estimate = (weights * lfc[:, None]).sum() / weights.sum()
        normalized_standard_error = np.sqrt((importance_weights / weights).sum(axis=0))

        tau2_samples, mean_samples = self.sample_tau2_likelihood(
            normalized_estimate, 
            normalized_standard_error.pow(2), 
            n_samples=5000, 
            prior_shape=1, 
            prior_rate=.1,
            mu_prior_mean=0, 
            mu_prior_var=0.01
        )
        # tau2 = np.mean(tau2_estimates[1000:])
        tau2 = self.mode_kde(tau2_samples[1000:])
        full_dataset_mean = mean_samples[1000:].mean()

        B = tau2 / (tau2 + normalized_standard_error.pow(2))

        shrunken_estimate = (
            B * normalized_estimate +
            (1 - B) * full_dataset_mean
        )

        shrunken_variance = (1 - B) * normalized_standard_error.pow(2)
        shrunken_standard_error = np.sqrt(shrunken_variance)

        table = pd.DataFrame({
            # Per Gene Statistics
            "weights": weights.sum(),
            "n_eff": (weights.sum().pow(2) / weights.pow(2).sum()),

            # Naive Estimates
            "naive_estimate": normalized_estimate,
            "naive_standard_error": normalized_standard_error,
            
            # EB Estimates
            "shrunken_estimate": shrunken_estimate,
            "shrunken_se": shrunken_standard_error,

            # Z-scores
            "z_naive": normalized_estimate / normalized_standard_error,
            "z_eb": shrunken_estimate / shrunken_standard_error,

            # pvalues
            "p_naive": norm.sf(np.abs(normalized_estimate / normalized_standard_error)) * 2,
            "p_eb": norm.sf(np.abs(shrunken_estimate / shrunken_standard_error)) * 2,

            # log10 for plotting
            "log10p_naive": -np.log10(norm.sf(np.abs(normalized_estimate / normalized_standard_error)) * 2),
            "log10p_eb": -np.log10(norm.sf(np.abs(shrunken_estimate / shrunken_standard_error)) * 2),

            # Full Dataset Statistics
            "B": B,
            "full_dataset_mean": full_dataset_mean,
            "full_dataset_standard_error": tau2,
        })

        return table

    @staticmethod
    def sample_tau2_likelihood(x, var, n_samples=5000, 
                            prior_shape=3, prior_rate=10,
                            mu_prior_mean=0, mu_prior_var=0.001):
        """
        Sample from posterior of tau^2 and mu using simple likelihood sampling.
        
        Parameters:
        -----------
        x : array
            Observed effect estimates
        var : array  
            Within-study variances
        n_samples : int
            Number of posterior samples
        prior_shape, prior_rate : float
            Inverse-Gamma(a, b) prior for tau^2 (shape=alpha, rate=beta)
            This prior keeps tau^2 > 0 always
        mu_prior_mean, mu_prior_var : float
            Normal prior for mu (overall mean)
        
        Returns:
        --------
        tau2_samples : array of shape (n_samples,)
        mu_samples : array of shape (n_samples,)
        """
        
        x = np.asarray(x)
        var = np.asarray(var)
        k = len(x)
        
        # Storage for samples
        tau2_samples = np.zeros(n_samples)
        mu_samples = np.zeros(n_samples)
        
        # Simple grid/metropolis approach (no MCMC libraries needed)
        # First, find reasonable range for tau^2
        var_obs = np.var(x)
        tau2_max = max(10, var_obs * 4)  # upper bound
        tau2_grid = np.linspace(1e-6, tau2_max, 500)
        
        # Pre-compute likelihood for each tau2
        log_weights = np.zeros(len(tau2_grid))
        
        for i, tau2 in enumerate(tau2_grid):
            # Marginal likelihood p(x | tau^2, mu) integrated over mu?
            # Actually easier: sample mu conditional on tau^2
            
            # Prior for tau^2
            log_prior = stats.invgamma.logpdf(tau2, a=prior_shape, scale=prior_rate)
            
            # For each tau2, we can compute the conditional posterior of mu
            # mu ~ N( (sum w_i x_i)/W, 1/W ) where w_i = 1/(var_i + tau^2)
            w = 1.0 / (var + tau2)
            W = w.sum()
            mu_cond_mean = (w * x).sum() / W
            mu_cond_var = 1.0 / W
            
            # Combine with prior on mu
            prec_prior = 1.0 / mu_prior_var
            prec_lik = W
            mu_post_var = 1.0 / (prec_prior + prec_lik)
            mu_post_mean = mu_post_var * (prec_prior * mu_prior_mean + prec_lik * mu_cond_mean)
            
            # Marginal log-likelihood (integrating out mu)
            # This is the normalizing constant for mu
            log_marginal = 0.5 * np.log(mu_post_var / mu_prior_var)
            log_marginal += 0.5 * (mu_cond_mean**2 / mu_cond_var - mu_post_mean**2 / mu_post_var)
            
            log_weights[i] = log_prior + log_marginal
        
        # Normalize weights
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / weights.sum()
        
        # Sample tau^2 from the grid
        tau2_idx = np.random.choice(len(tau2_grid), size=n_samples, p=weights)
        tau2_samples = tau2_grid[tau2_idx]
        
        # For each sampled tau^2, sample mu from its conditional posterior
        for i, tau2 in enumerate(tau2_samples):
            w = 1.0 / (var + tau2)
            W = w.sum()
            mu_cond_mean = (w * x).sum() / W
            mu_cond_var = 1.0 / W
            
            # Combine with prior
            prec_prior = 1.0 / mu_prior_var
            prec_lik = W
            mu_post_var = 1.0 / (prec_prior + prec_lik)
            mu_post_mean = mu_post_var * (prec_prior * mu_prior_mean + prec_lik * mu_cond_mean)
            
            mu_samples[i] = np.random.normal(mu_post_mean, np.sqrt(mu_post_var))
        
        return tau2_samples, mu_samples

    @staticmethod
    def mode_kde(samples, grid_points=200):
        """
        Approximate mode using kernel density estimation.
        More accurate than histogram.
        """
        kde = gaussian_kde(samples)
        grid = np.linspace(min(samples), max(samples), grid_points)
        density = kde.evaluate(grid)
        mode = grid[np.argmax(density)]
        return mode