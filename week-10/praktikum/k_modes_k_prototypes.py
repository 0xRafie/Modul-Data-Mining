"""
Demo K-Modes dan K-Prototypes untuk Week 10.

Dataset umum yang dipakai:
1. Breast Cancer Wisconsin dari scikit-learn, didiskritisasi menjadi kategori
   Low/Medium/High untuk demo K-Modes.
2. Palmer Penguins dari seaborn-data untuk demo K-Prototypes mixed data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


class SimpleKModes:
    """Minimal K-Modes implementation for teaching purposes."""

    def __init__(self, n_clusters: int, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centroids_: np.ndarray | None = None
        self.cost_: float | None = None

    @staticmethod
    def _mode(values: np.ndarray) -> object:
        unique, counts = np.unique(values, return_counts=True)
        return unique[np.argmax(counts)]

    @staticmethod
    def _matching_distance(data: np.ndarray, modes: np.ndarray) -> np.ndarray:
        return np.array([(data != mode).sum(axis=1) for mode in modes]).T

    def fit_predict(self, data: pd.DataFrame) -> np.ndarray:
        X = data.to_numpy(dtype=object)
        rng = np.random.default_rng(self.random_state)
        modes = X[rng.choice(len(X), size=self.n_clusters, replace=False)].copy()
        labels = np.zeros(len(X), dtype=int)

        for _ in range(self.max_iter):
            new_labels = self._matching_distance(X, modes).argmin(axis=1)
            new_modes = modes.copy()

            for cluster_id in range(self.n_clusters):
                members = X[new_labels == cluster_id]
                if len(members) == 0:
                    new_modes[cluster_id] = X[rng.integers(0, len(X))]
                    continue
                for col_idx in range(X.shape[1]):
                    new_modes[cluster_id, col_idx] = self._mode(members[:, col_idx])

            if np.array_equal(new_labels, labels) and np.array_equal(new_modes, modes):
                break
            labels = new_labels
            modes = new_modes

        self.cluster_centroids_ = modes
        self.cost_ = float(self._matching_distance(X, modes).min(axis=1).sum())
        return labels


class SimpleKPrototypes:
    """Minimal K-Prototypes implementation for teaching purposes."""

    def __init__(
        self,
        n_clusters: int,
        gamma: float = 1.0,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centroids_: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.cost_: float | None = None

    @staticmethod
    def _mode(values: np.ndarray) -> object:
        unique, counts = np.unique(values, return_counts=True)
        return unique[np.argmax(counts)]

    def _distance(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        numeric_centers: np.ndarray,
        categorical_modes: np.ndarray,
    ) -> np.ndarray:
        distances = []
        for num_center, cat_mode in zip(numeric_centers, categorical_modes):
            numeric_cost = ((numeric - num_center) ** 2).sum(axis=1)
            categorical_cost = (categorical != cat_mode).sum(axis=1)
            distances.append(numeric_cost + self.gamma * categorical_cost)
        return np.array(distances).T

    def fit_predict(self, numeric: np.ndarray, categorical: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        initial_idx = rng.choice(len(numeric), size=self.n_clusters, replace=False)
        numeric_centers = numeric[initial_idx].copy()
        categorical_modes = categorical[initial_idx].copy()
        labels = np.zeros(len(numeric), dtype=int)

        for _ in range(self.max_iter):
            distances = self._distance(numeric, categorical, numeric_centers, categorical_modes)
            new_labels = distances.argmin(axis=1)

            new_numeric_centers = numeric_centers.copy()
            new_categorical_modes = categorical_modes.copy()
            for cluster_id in range(self.n_clusters):
                members_num = numeric[new_labels == cluster_id]
                members_cat = categorical[new_labels == cluster_id]
                if len(members_num) == 0:
                    replacement = rng.integers(0, len(numeric))
                    new_numeric_centers[cluster_id] = numeric[replacement]
                    new_categorical_modes[cluster_id] = categorical[replacement]
                    continue
                new_numeric_centers[cluster_id] = members_num.mean(axis=0)
                for col_idx in range(categorical.shape[1]):
                    new_categorical_modes[cluster_id, col_idx] = self._mode(
                        members_cat[:, col_idx]
                    )

            if (
                np.array_equal(new_labels, labels)
                and np.allclose(new_numeric_centers, numeric_centers)
                and np.array_equal(new_categorical_modes, categorical_modes)
            ):
                break

            labels = new_labels
            numeric_centers = new_numeric_centers
            categorical_modes = new_categorical_modes

        self.cluster_centroids_ = [
            (numeric_centers[i], categorical_modes[i]) for i in range(self.n_clusters)
        ]
        self.cost_ = float(
            self._distance(numeric, categorical, numeric_centers, categorical_modes)
            .min(axis=1)
            .sum()
        )
        return labels


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def plot_breast_cancer_kmodes(original_numeric: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(StandardScaler().fit_transform(original_numeric))

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=30, alpha=0.8)
    ax.set_title("K-Modes on Discretized Breast Cancer Dataset")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_kmodes_breast_cancer_pca.png", dpi=150)
    plt.close()


def run_kmodes_demo() -> None:
    print_section("1. K-MODES: BREAST CANCER WISCONSIN DISCRETIZED")

    cancer = load_breast_cancer(as_frame=True)
    numeric = cancer.data

    selected_cols = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "worst radius",
        "worst texture",
        "worst perimeter",
    ]
    selected = numeric[selected_cols]

    discretizer = KBinsDiscretizer(
        n_bins=3,
        encode="ordinal",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
    )
    discretized = discretizer.fit_transform(selected).astype(int)
    category_labels = np.array(["Low", "Medium", "High"])
    categorical = pd.DataFrame(category_labels[discretized], columns=selected_cols)

    kmodes = SimpleKModes(n_clusters=2, random_state=42)
    labels = kmodes.fit_predict(categorical)

    result = categorical.copy()
    result["cluster"] = labels
    result["target"] = cancer.target_names[cancer.target]

    print("Dataset: Breast Cancer Wisconsin (scikit-learn)")
    print("Fitur numeric didiskritisasi menjadi Low/Medium/High untuk demo K-Modes.")
    print("\nContoh data kategorikal:")
    print(result.head())
    print("\nCluster modes:")
    print(pd.DataFrame(kmodes.cluster_centroids_, columns=selected_cols))
    print("\nCrosstab cluster vs diagnosis asli:")
    print(pd.crosstab(result["cluster"], result["target"]))
    print(f"\nK-Modes cost: {kmodes.cost_:.2f}")

    plot_breast_cancer_kmodes(selected.to_numpy(), labels)


def load_penguins() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    penguins = pd.read_csv(url)
    needed = [
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]
    return penguins[needed].dropna().reset_index(drop=True)


def summarize_kprototypes(
    data: pd.DataFrame, labels: np.ndarray, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    rows = []
    for cluster_id in sorted(np.unique(labels)):
        subset = data.loc[labels == cluster_id]
        row = {"cluster": cluster_id, "size": len(subset)}
        for col in numeric_cols:
            row[f"mean_{col}"] = round(float(subset[col].mean()), 2)
        for col in categorical_cols:
            row[f"mode_{col}"] = subset[col].mode(dropna=True).iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_penguins_kprototypes(data: pd.DataFrame, labels: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        data["bill_length_mm"],
        data["flipper_length_mm"],
        c=labels,
        cmap="tab10",
        s=55,
        alpha=0.85,
        edgecolors="black",
    )
    ax.set_title("K-Prototypes on Palmer Penguins")
    ax.set_xlabel("Bill Length (mm)")
    ax.set_ylabel("Flipper Length (mm)")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_kprototypes_penguins.png", dpi=150)
    plt.close()


def run_kprototypes_demo() -> None:
    print_section("2. K-PROTOTYPES: PALMER PENGUINS MIXED DATA")

    penguins = load_penguins()
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_cols = ["island", "sex"]

    scaler = StandardScaler()
    numeric = scaler.fit_transform(penguins[numeric_cols])
    categorical = penguins[categorical_cols].to_numpy(dtype=object)

    kproto = SimpleKPrototypes(n_clusters=3, gamma=1.0, random_state=42)
    labels = kproto.fit_predict(numeric, categorical)

    result = penguins.copy()
    result["cluster"] = labels

    print("Dataset: Palmer Penguins (seaborn-data)")
    print("Numeric: bill/flipper/body measurements; Categorical: island, sex.")
    print("\nContoh data:")
    print(result.head())
    print("\nCluster profile:")
    print(summarize_kprototypes(penguins, labels, numeric_cols, categorical_cols))
    print("\nCrosstab cluster vs species asli:")
    print(pd.crosstab(result["cluster"], result["species"]))
    print(f"\nK-Prototypes cost: {kproto.cost_:.2f}")

    gamma_rows = []
    for gamma in [0.2, 1.0, 3.0, 6.0]:
        model = SimpleKPrototypes(n_clusters=3, gamma=gamma, random_state=42)
        gamma_labels = model.fit_predict(numeric, categorical)
        gamma_rows.append(
            {
                "gamma": gamma,
                "cost": round(float(model.cost_), 2),
                "cluster_sizes": pd.Series(gamma_labels).value_counts().sort_index().to_dict(),
            }
        )

    print("\nGamma sensitivity:")
    print(pd.DataFrame(gamma_rows))

    plot_penguins_kprototypes(penguins, labels)


def main() -> None:
    run_kmodes_demo()
    run_kprototypes_demo()
    print_section("OUTPUT")
    print(f"Figures tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
