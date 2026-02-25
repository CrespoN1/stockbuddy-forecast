"""
PCA (Principal Component Analysis) module for StockBuddy Forecast.

Performs dimensionality reduction on the multi-feature stock dataset,
analyzes explained variance, and provides visualization utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List

from src.config import PCA_VARIANCE_THRESHOLD, FIGURES_DIR


class StockPCA:
    """
    PCA analysis wrapper for stock feature data.

    Handles standardization, fitting, transformation, and analysis
    of principal components.
    """

    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = PCA_VARIANCE_THRESHOLD):
        """
        Args:
            n_components: Fixed number of components (None = auto-select)
            variance_threshold: Minimum cumulative variance to explain (if n_components is None)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names: List[str] = []
        self.ticker_names: List[str] = []
        self.is_fitted = False

    def fit_transform(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and apply PCA to the feature matrix.

        Args:
            feature_matrix: DataFrame with tickers as index, features as columns

        Returns:
            DataFrame with tickers as index, PC1, PC2, ... as columns
        """
        self.feature_names = list(feature_matrix.columns)
        self.ticker_names = list(feature_matrix.index)

        # Standardize features
        X_scaled = self.scaler.fit_transform(feature_matrix.values)

        # First fit with all components to analyze variance
        pca_full = PCA()
        pca_full.fit(X_scaled)

        # Determine number of components
        if self.n_components is None:
            cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = int(np.argmax(cumulative_var >= self.variance_threshold) + 1)
            self.n_components = max(2, self.n_components)  # At least 2 for visualization
            print(f"Auto-selected {self.n_components} components "
                  f"(explaining {cumulative_var[self.n_components-1]:.1%} variance)")

        # Fit final PCA
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        # Create result DataFrame
        columns = [f"PC{i+1}" for i in range(self.n_components)]
        result = pd.DataFrame(X_pca, index=feature_matrix.index, columns=columns)

        self.is_fitted = True
        return result

    def transform(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted PCA."""
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet. Call fit_transform first.")

        X_scaled = self.scaler.transform(feature_matrix.values)
        X_pca = self.pca.transform(X_scaled)

        columns = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(X_pca, index=feature_matrix.index, columns=columns)

    def get_explained_variance(self) -> pd.DataFrame:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet.")

        return pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(self.n_components)],
            "Explained_Variance": self.pca.explained_variance_ratio_,
            "Cumulative_Variance": np.cumsum(self.pca.explained_variance_ratio_),
        })

    def get_loadings(self) -> pd.DataFrame:
        """
        Get PCA loadings (feature contributions to each component).

        Returns:
            DataFrame with features as index, PCs as columns
        """
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet.")

        columns = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=columns,
        )

    def get_top_features(self, component: int = 1, n_top: int = 10) -> pd.DataFrame:
        """
        Get the top contributing features for a given principal component.
        """
        loadings = self.get_loadings()
        pc_col = f"PC{component}"

        top = loadings[pc_col].abs().sort_values(ascending=False).head(n_top)

        result = pd.DataFrame({
            "Feature": top.index,
            "Loading": loadings.loc[top.index, pc_col].values,
            "Abs_Loading": top.values,
        })

        return result

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_explained_variance(self, save: bool = False, filename: str = "pca_variance.png"):
        """Plot explained variance ratio (scree plot)."""
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet.")

        var_df = self.get_explained_variance()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax1.bar(var_df["Component"], var_df["Explained_Variance"], color="steelblue", alpha=0.8)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("Individual Explained Variance")
        ax1.tick_params(axis="x", rotation=45)

        # Cumulative variance
        ax2.plot(var_df["Component"], var_df["Cumulative_Variance"],
                 "o-", color="steelblue", linewidth=2)
        ax2.axhline(y=self.variance_threshold, color="red", linestyle="--",
                     label=f"Threshold ({self.variance_threshold:.0%})")
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("Cumulative Explained Variance")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()

    def plot_2d_scatter(
        self,
        pca_data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        title: str = "Stocks in PCA Space",
        save: bool = False,
        filename: str = "pca_scatter.png",
    ):
        """
        Plot stocks in 2D PCA space.

        Args:
            pca_data: Output from fit_transform
            labels: Optional cluster labels or sector labels for coloring
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        if labels is not None:
            unique_labels = labels.unique()
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))

            for i, label in enumerate(sorted(unique_labels)):
                mask = labels == label
                ax.scatter(
                    pca_data.loc[mask, "PC1"],
                    pca_data.loc[mask, "PC2"],
                    c=[cmap(i)],
                    label=str(label),
                    alpha=0.7,
                    s=50,
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        else:
            ax.scatter(pca_data["PC1"], pca_data["PC2"], alpha=0.7, s=50, c="steelblue")

        # Annotate points with ticker names
        for ticker in pca_data.index:
            ax.annotate(
                ticker,
                (pca_data.loc[ticker, "PC1"], pca_data.loc[ticker, "PC2"]),
                fontsize=6,
                alpha=0.6,
            )

        ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()

    def plot_loadings_heatmap(
        self,
        n_features: int = 20,
        n_components: int = 5,
        save: bool = False,
        filename: str = "pca_loadings.png",
    ):
        """Plot heatmap of top feature loadings."""
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet.")

        loadings = self.get_loadings()

        # Select top features by max absolute loading across components
        n_comp = min(n_components, self.n_components)
        top_cols = loadings.iloc[:, :n_comp]
        max_abs = top_cols.abs().max(axis=1).sort_values(ascending=False)
        top_features = max_abs.head(n_features).index

        plot_data = loadings.loc[top_features, loadings.columns[:n_comp]]

        fig, ax = plt.subplots(figsize=(10, max(8, n_features * 0.4)))
        sns.heatmap(
            plot_data,
            center=0,
            cmap="RdBu_r",
            annot=True,
            fmt=".2f",
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("PCA Feature Loadings (Top Features)")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Feature")

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()
