import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

class ForestPlotter:

    def __init__(self, deseq_results, balanced_matrix, gene_ontology_results):
        self.deseq_results = deseq_results
        self.balanced_matrix = balanced_matrix
        self.gene_ontology_results = gene_ontology_results

    def plot_term(
        self,
        term: str,
        cutoff=0.99,
        minimum_kept=5,
        figsize=None,
        ci_multiplier=1.96,
        point_size=45,
        contribution_label=True,
        title=None,
    ):
        contributing_genes = self.balanced_matrix.index[
            self.balanced_matrix[term].gt(0)
        ]

        subset_df = self.deseq_results.loc[contributing_genes].copy()
        subset_weights = self.balanced_matrix.loc[contributing_genes, term]

        contribution = subset_weights / subset_weights.sum()
        meta_estimation = self.gene_ontology_results.loc[term]

        forest_df = pd.DataFrame({
            "gene": subset_df.index,
            "L2FC": subset_df["log2FoldChange"],
            "SE": subset_df["lfcSE"],
            "lower_ci": subset_df["log2FoldChange"] - ci_multiplier * subset_df["lfcSE"],
            "upper_ci": subset_df["log2FoldChange"] + ci_multiplier * subset_df["lfcSE"],
            "pct_contribution": contribution,
        })

        forest_df = forest_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["L2FC", "SE", "pct_contribution"]
        )

        if cutoff is not None:
            keep_genes = (
                forest_df
                .sort_values("pct_contribution", ascending=False)
                .assign(cum_contribution=lambda df: df["pct_contribution"].cumsum())
                .loc[lambda df: df["cum_contribution"].le(cutoff), "gene"]
            )

            if len(keep_genes) <= minimum_kept:
                keep_genes = (
                    forest_df
                    .sort_values("pct_contribution", ascending=False)
                    .head(minimum_kept)["gene"]
                )

            forest_df = forest_df.loc[forest_df["gene"].isin(keep_genes)]

        forest_df = forest_df.sort_values("pct_contribution", ascending=True)

        n_genes = len(forest_df)

        if figsize is None:
            figsize = (11, max(5, 0.32 * n_genes + 2.5))

        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={
                "width_ratios": [2.4, 1],
                "height_ratios": [max(4, n_genes * 0.3), 1],
                "hspace": 0.2,
                "wspace": 0.05,
            },
            figsize=figsize,
            sharex="col",
        )

        forest_ax = ax[0, 0]
        contrib_ax = ax[0, 1]
        meta_ax = ax[1, 0]
        empty_ax = ax[1, 1]

        y = np.arange(n_genes)

        colors = np.where(forest_df["L2FC"].ge(0), "firebrick", "steelblue")

        forest_ax.errorbar(
            x=forest_df["L2FC"],
            y=y,
            xerr=ci_multiplier * forest_df["SE"],
            fmt="none",
            ecolor="0.35",
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )

        forest_ax.scatter(
            x=forest_df["L2FC"],
            y=y,
            s=point_size,
            c=colors,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

        forest_ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        forest_ax.set_yticks(y)
        forest_ax.set_yticklabels(forest_df["gene"])
        forest_ax.set_ylabel("")
        forest_ax.set_xlabel("log2 fold change")
        forest_ax.grid(axis="x", alpha=0.25)
        forest_ax.spines["top"].set_visible(False)
        forest_ax.spines["right"].set_visible(False)

        contrib_ax.barh(
            y=y,
            width=forest_df["pct_contribution"],
            color="0.45",
            edgecolor="black",
            linewidth=0.3,
        )

        contrib_ax.set_yticks(y)
        contrib_ax.set_yticklabels([])
        contrib_ax.set_xlabel("Contribution")
        contrib_ax.grid(axis="x", alpha=0.25)
        contrib_ax.spines["top"].set_visible(False)
        contrib_ax.spines["right"].set_visible(False)
        contrib_ax.spines["left"].set_visible(False)
        contrib_ax.tick_params(axis="y", length=0)

        if contribution_label:
            xmax = forest_df["pct_contribution"].max()
            for yi, value in zip(y, forest_df["pct_contribution"]):
                contrib_ax.text(
                    value + xmax * 0.02,
                    yi,
                    f"{value:.1%}",
                    va="center",
                    fontsize=8,
                )
            contrib_ax.set_xlim(0, xmax * 1.25)

        meta_lfc = meta_estimation["shrunken_estimate"]
        meta_se = meta_estimation["shrunken_se"]

        meta_ax.errorbar(
            x=meta_lfc,
            y=[0],
            xerr=ci_multiplier * meta_se,
            fmt="none",
            ecolor="0.35",
            elinewidth=1.4,
            capsize=4,
            zorder=1,
        )

        meta_color = "firebrick" if meta_lfc >= 0 else "steelblue"

        meta_ax.scatter(
            x=[meta_lfc],
            y=[0],
            s=70,
            c=meta_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

        meta_ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        meta_ax.set_yticks([0])
        meta_ax.set_yticklabels(["Meta-estimate"])
        meta_ax.set_xlabel("meta log2 fold change")
        meta_ax.grid(axis="x", alpha=0.25)
        meta_ax.spines["top"].set_visible(False)
        meta_ax.spines["right"].set_visible(False)

        empty_ax.axis("off")

        if title is None:
            title = term

        fig.suptitle(
            title,
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        displayed_contribution = forest_df["pct_contribution"].sum()

        fig.text(
            0.01,
            0.01,
            f"Displayed genes: {n_genes} | Displayed contribution: {displayed_contribution:.1%}",
            fontsize=9,
            color="0.35",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        return forest_df, fig, ax

def volcano_plot(
    table,
    lfc_col="shrunken_estimate",
    pval_col="log10p_eb",
    gene_col=None,
    size_col="n_eff",
    lfc_thresh=1.0,
    pval_thresh=2.0,
    x_label="Log2 Fold Change",
    y_label="-log10(p-value)",
    title="Volcano Plot",
    up_color="#d62728",      # red
    down_color="#1f77b4",    # blue
    ns_color="#bdbdbd",      # grey
    width=1000,
    height=750,
    size_max=40,
):
    """
    Create a customizable volcano plot with Plotly.

    Parameters
    ----------
    table : pd.DataFrame
        Input dataframe.
    lfc_col : str
        Column containing log fold changes.
    pval_col : str
        Column containing -log10(p-values).
    gene_col : str or None
        Column to use for hover labels. If None, uses index.
    size_col : str or None
        Column controlling point size.
    lfc_thresh : float
        Absolute log fold change threshold.
    pval_thresh : float
        -log10(p-value) threshold.
    """

    df = table.copy()

    # Hover labels
    if gene_col is None:
        df["_gene_label"] = df.index.astype(str)
        gene_col = "_gene_label"

    # Classification
    conditions = [
        (df[lfc_col] >= lfc_thresh) & (df[pval_col] >= pval_thresh),
        (df[lfc_col] <= -lfc_thresh) & (df[pval_col] >= pval_thresh),
    ]

    choices = ["Increasing", "Decreasing"]

    df["_category"] = np.select(conditions, choices, default="NS")

    color_map = {
        "Increasing": up_color,
        "Decreasing": down_color,
        "NS": ns_color,
    }

    # Plot
    fig = px.scatter(
        df,
        x=lfc_col,
        y=pval_col,
        color="_category",
        color_discrete_map=color_map,
        size=size_col,
        hover_name=gene_col,
        hover_data={
            lfc_col: True,
            pval_col: True,
            size_col: True if size_col else False,
            "_category": False,
        },
        width=width,
        height=height,
        size_max=size_max,
    )

    # Threshold lines
    fig.add_vline(
        x=lfc_thresh,
        line_dash="dash",
        line_color="black"
    )

    fig.add_vline(
        x=-lfc_thresh,
        line_dash="dash",
        line_color="black"
    )

    fig.add_hline(
        y=pval_thresh,
        line_dash="dash",
        line_color="black"
    )

    # White background
    fig.update_layout(
        template="simple_white",
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Category",
    )

    # Axis labels
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)

    return fig