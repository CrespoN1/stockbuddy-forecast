"""
Stock clustering module for StockBuddy Forecast.

Implements K-Means and Gaussian Mixture Model clustering on PCA-reduced
stock features, with methods for optimal cluster selection and visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Dict, List, Optional, Tuple

from src.config import MAX_CLUSTERS, RANDOM_STATE, FIGURES_DIR


class StockClusterer:
    """
    Stock clustering using K-Means and Gaussian Mixture Models.
    """

    def __init__(self, max_clusters: int = MAX_CLUSTERS, random_state: int = RANDOM_STATE):
        self.max_clusters = max_clusters
        self.random_state = random_state

        self.kmeans_model = None
        self.gmm_model = None
        self.optimal_k_kmeans: Optional[int] = None
        self.optimal_k_gmm: Optional[int] = None

        # Store evaluation metrics
        self.kmeans_metrics: Dict = {}
        self.gmm_metrics: Dict = {}

    # =========================================================================
    # K-Means Clustering
    # =========================================================================

    def evaluate_kmeans(self, X: pd.DataFrame, k_range: Optional[range] = None) -> pd.DataFrame:
        """
        Evaluate K-Means for different values of k.

        Returns DataFrame with inertia, silhouette scores for each k.
        """
        if k_range is None:
            k_range = range(2, min(self.max_clusters + 1, len(X)))

        results = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)

            inertia = km.inertia_
            sil_score = silhouette_score(X, labels) if k > 1 else 0

            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette": sil_score,
            })

        self.kmeans_metrics = pd.DataFrame(results)

        # Auto-select optimal k based on silhouette score
        self.optimal_k_kmeans = int(
            self.kmeans_metrics.loc[self.kmeans_metrics["silhouette"].idxmax(), "k"]
        )
        print(f"Optimal K-Means k = {self.optimal_k_kmeans} (silhouette = "
              f"{self.kmeans_metrics['silhouette'].max():.3f})")

        return self.kmeans_metrics

    def fit_kmeans(self, X: pd.DataFrame, k: Optional[int] = None) -> pd.Series:
        """
        Fit K-Means with specified or optimal k.

        Returns:
            Series of cluster labels indexed by ticker
        """
        if k is None:
            if self.optimal_k_kmeans is None:
                self.evaluate_kmeans(X)
            k = self.optimal_k_kmeans

        self.kmeans_model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = self.kmeans_model.fit_predict(X)

        return pd.Series(labels, index=X.index, name="KMeans_Cluster")

    # =========================================================================
    # Gaussian Mixture Model Clustering
    # =========================================================================

    def evaluate_gmm(self, X: pd.DataFrame, k_range: Optional[range] = None) -> pd.DataFrame:
        """
        Evaluate GMM for different values of k using BIC and AIC.
        """
        if k_range is None:
            k_range = range(2, min(self.max_clusters + 1, len(X)))

        results = []

        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                random_state=self.random_state,
                covariance_type="full",
                n_init=5,
            )
            gmm.fit(X)
            labels = gmm.predict(X)

            bic = gmm.bic(X)
            aic = gmm.aic(X)
            sil_score = silhouette_score(X, labels) if k > 1 and len(set(labels)) > 1 else 0

            results.append({
                "k": k,
                "bic": bic,
                "aic": aic,
                "silhouette": sil_score,
            })

        self.gmm_metrics = pd.DataFrame(results)

        # Auto-select optimal k based on BIC (lower is better)
        self.optimal_k_gmm = int(
            self.gmm_metrics.loc[self.gmm_metrics["bic"].idxmin(), "k"]
        )
        print(f"Optimal GMM k = {self.optimal_k_gmm} (BIC = "
              f"{self.gmm_metrics['bic'].min():.1f})")

        return self.gmm_metrics

    def fit_gmm(self, X: pd.DataFrame, k: Optional[int] = None) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Fit GMM with specified or optimal k.

        Returns:
            Tuple of (hard cluster labels, soft probability matrix)
        """
        if k is None:
            if self.optimal_k_gmm is None:
                self.evaluate_gmm(X)
            k = self.optimal_k_gmm

        self.gmm_model = GaussianMixture(
            n_components=k,
            random_state=self.random_state,
            covariance_type="full",
            n_init=5,
        )
        self.gmm_model.fit(X)

        labels = self.gmm_model.predict(X)
        probabilities = self.gmm_model.predict_proba(X)

        labels_series = pd.Series(labels, index=X.index, name="GMM_Cluster")
        prob_df = pd.DataFrame(
            probabilities,
            index=X.index,
            columns=[f"GMM_Prob_C{i}" for i in range(k)],
        )

        return labels_series, prob_df

    # =========================================================================
    # Cluster Analysis
    # =========================================================================

    def get_cluster_summary(
        self,
        labels: pd.Series,
        feature_matrix: pd.DataFrame,
        sector_info: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate summary statistics for each cluster.
        """
        summary_data = feature_matrix.copy()
        summary_data["Cluster"] = labels

        if sector_info is not None:
            summary_data["Sector"] = sector_info

        # Basic cluster stats
        cluster_stats = summary_data.groupby("Cluster").agg(
            count=("Cluster", "size"),
        )

        # Add sector composition if available
        if sector_info is not None:
            sector_counts = summary_data.groupby(["Cluster", "Sector"]).size().unstack(fill_value=0)
            # Get dominant sector per cluster
            cluster_stats["Dominant_Sector"] = sector_counts.idxmax(axis=1)
            cluster_stats["Sector_Concentration"] = sector_counts.max(axis=1) / sector_counts.sum(axis=1)

        # Add tickers per cluster
        cluster_stats["Tickers"] = summary_data.groupby("Cluster").apply(
            lambda x: ", ".join(x.index[:10])  # First 10 tickers
        )

        return cluster_stats

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_elbow_and_silhouette(self, save: bool = False, filename: str = "kmeans_evaluation.png"):
        """Plot K-Means elbow curve and silhouette scores."""
        if self.kmeans_metrics is None or len(self.kmeans_metrics) == 0:
            raise ValueError("Run evaluate_kmeans first.")

        metrics = self.kmeans_metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow curve
        ax1.plot(metrics["k"], metrics["inertia"], "bo-", linewidth=2)
        if self.optimal_k_kmeans:
            mask = metrics["k"] == self.optimal_k_kmeans
            ax1.axvline(x=self.optimal_k_kmeans, color="red", linestyle="--",
                        label=f"Optimal k={self.optimal_k_kmeans}")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia (Within-Cluster SSE)")
        ax1.set_title("K-Means Elbow Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette scores
        ax2.plot(metrics["k"], metrics["silhouette"], "ro-", linewidth=2)
        if self.optimal_k_kmeans:
            ax2.axvline(x=self.optimal_k_kmeans, color="blue", linestyle="--",
                        label=f"Optimal k={self.optimal_k_kmeans}")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("K-Means Silhouette Scores")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()

    def plot_gmm_evaluation(self, save: bool = False, filename: str = "gmm_evaluation.png"):
        """Plot GMM BIC/AIC curves."""
        if self.gmm_metrics is None or len(self.gmm_metrics) == 0:
            raise ValueError("Run evaluate_gmm first.")

        metrics = self.gmm_metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # BIC and AIC
        ax1.plot(metrics["k"], metrics["bic"], "bo-", linewidth=2, label="BIC")
        ax1.plot(metrics["k"], metrics["aic"], "rs-", linewidth=2, label="AIC")
        if self.optimal_k_gmm:
            ax1.axvline(x=self.optimal_k_gmm, color="green", linestyle="--",
                        label=f"Optimal k={self.optimal_k_gmm}")
        ax1.set_xlabel("Number of Components (k)")
        ax1.set_ylabel("Information Criterion")
        ax1.set_title("GMM Model Selection")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette scores
        ax2.plot(metrics["k"], metrics["silhouette"], "go-", linewidth=2)
        if self.optimal_k_gmm:
            ax2.axvline(x=self.optimal_k_gmm, color="blue", linestyle="--",
                        label=f"Optimal k={self.optimal_k_gmm}")
        ax2.set_xlabel("Number of Components (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("GMM Silhouette Scores")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()

    def plot_cluster_scatter(
        self,
        pca_data: pd.DataFrame,
        labels: pd.Series,
        title: str = "Stock Clusters in PCA Space",
        save: bool = False,
        filename: str = "cluster_scatter.png",
    ):
        """Plot clusters in 2D PCA space."""
        fig, ax = plt.subplots(figsize=(12, 8))

        unique_labels = sorted(labels.unique())
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                pca_data.loc[mask, "PC1"],
                pca_data.loc[mask, "PC2"],
                c=[cmap(i)],
                label=f"Cluster {label}",
                alpha=0.7,
                s=60,
                edgecolors="white",
                linewidths=0.5,
            )

        # Annotate with ticker names
        for ticker in pca_data.index:
            ax.annotate(
                ticker,
                (pca_data.loc[ticker, "PC1"], pca_data.loc[ticker, "PC2"]),
                fontsize=6,
                alpha=0.5,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()

    def plot_cluster_composition(
        self,
        labels: pd.Series,
        sector_info: pd.Series,
        save: bool = False,
        filename: str = "cluster_composition.png",
    ):
        """Plot sector composition of each cluster as a stacked bar chart."""
        cross_tab = pd.crosstab(labels, sector_info, normalize="index")

        fig, ax = plt.subplots(figsize=(12, 6))
        cross_tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Proportion")
        ax.set_title("Sector Composition by Cluster")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout()

        if save:
            import os
            os.makedirs(FIGURES_DIR, exist_ok=True)
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

        plt.show()
