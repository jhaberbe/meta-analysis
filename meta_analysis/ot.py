import numpy as np
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

class IterativeProportionalFitting:

    @staticmethod
    def get_balanced_matrix(
        assignment_matrix: pd.DataFrame, 
        max_iter = 1000,
        tolerance = 1e-3,
        nonsquare_preference = "columns"
    ):
        # FIXME: I feel like this is bad form generally, to strip off the indices.
        A = assignment_matrix.values
        u = np.ones(A.shape[0])
        v = np.ones(A.shape[1])

        delta = 1

        with tqdm(range(max_iter), desc="IPF Running") as pbar:
            for _ in pbar:
                # Rescale
                A_u1 = ((A * u[:, None]) / A.sum(axis=0))
                A_u2 = ((A_u1 * v) / A_u1.sum(axis=0))

                # Compute % change in Frobenius Norm
                delta = np.pow((A - A_u2), 2).sum() / np.pow(A, 2).sum()
                pbar.set_postfix({"% Change (Frob Norm)": delta})

                # Update matrix
                A = A_u2

                if delta < tolerance:
                    return pd.DataFrame(
                        A,
                        index=assignment_matrix.index,
                        columns=assignment_matrix.columns
                    )

        return pd.DataFrame(
            A,
            index=assignment_matrix.index,
            columns=assignment_matrix.columns
        )

class UnbalancedSinkhorn:

    @staticmethod
    def get_balanced_matrix(
        assignment_matrix: pd.DataFrame,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        tau: float = 0.5,
        epsilon: float = 1e-12,
    ) -> pd.DataFrame:
        """
        Unbalanced Sinkhorn / IPF on a sparse binary support matrix.

        Parameters
        ----------
        assignment_matrix : pd.DataFrame
            Binary or nonnegative support matrix.

        max_iter : int
            Maximum iterations.

        tolerance : float
            Convergence threshold.

        tau : float
            Marginal relaxation parameter.

            tau = 1.0  -> classical Sinkhorn
            tau < 1.0 -> relaxed / unbalanced OT

        epsilon : float
            Small numerical stabilizer.

        Returns
        -------
        pd.DataFrame
            Balanced transport / assignment matrix.
        """

        # Original support mask
        support = (assignment_matrix.values > 0)

        # Float matrix
        A = assignment_matrix.values.astype(float)

        # Numerical stabilization
        A = A + epsilon

        m, n = A.shape

        # Degree-aware marginals
        # Prevents sparse rows/cols from exploding
        r = A.sum(axis=1)
        r = r / r.sum()

        c = A.sum(axis=0)
        c = c / c.sum()

        # Scaling vectors
        u = np.ones(m)
        v = np.ones(n)

        for _ in tqdm(range(max_iter), desc="Unbalanced Sinkhorn"):

            u_prev = u.copy()

            # Row update (relaxed)
            Kv = A @ v
            Kv = np.maximum(Kv, epsilon)

            u = (r / Kv) ** tau

            # Column update (relaxed)
            KTu = A.T @ u
            KTu = np.maximum(KTu, epsilon)

            v = (c / KTu) ** tau

            # Construct transport plan
            P = (u[:, None] * A) * v[None, :]

            # STRICT support preservation
            # zeros remain zero
            P *= support

            # Convergence
            delta = np.linalg.norm(u - u_prev)

            if delta < tolerance:
                break

        return pd.DataFrame(
            P,
            index=assignment_matrix.index,
            columns=assignment_matrix.columns
        )