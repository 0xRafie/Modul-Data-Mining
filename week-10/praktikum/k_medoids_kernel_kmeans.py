"""
Demo K-Medoids dan kernel/spectral clustering untuk Week 10.

Dataset umum yang dipakai:
1. Iris dari scikit-learn untuk K-Medoids vs K-Means.
2. Digits dari scikit-learn untuk K-Means vs SpectralClustering RBF.
3. Noisy circles tetap ditambahkan sebagai benchmark klasik non-linear clustering.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import load_digits, load_iris, make_circles
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


class SimpleKMedoids:
    """Small K-Medoids implementation for teaching purposes."""

    def __init__(self, n_clusters: int, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float | None = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        distances = pairwise_distances(X, metric="euclidean")
        medoid_indices = rng.choice(len(X), size=self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            labels = distances[:, medoid_indices].argmin(axis=1)
            new_medoids = medoid_indices.copy()

            for cluster_id in range(self.n_clusters):
                members = np.where(labels == cluster_id)[0]
                if len(members) == 0:
                    new_medoids[cluster_id] = rng.integers(0, len(X))
                    continue
                intra_distances = distances[np.ix_(members, members)]
                best_member_position = intra_distances.sum(axis=1).argmin()
                new_medoids[cluster_id] = members[best_member_position]

            if np.array_equal(new_medoids, medoid_indices):
                break
            medoid_indices = new_medoids

        labels = distances[:, medoid_indices].argmin(axis=1)
        self.medoid_indices_ = medoid_indices
        self.cluster_centers_ = X[medoid_indices]
        self.inertia_ = float(distances[:, medoid_indices].min(axis=1).sum())
        return labels


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def plot_iris_comparison(
    coords: np.ndarray,
    y_true: np.ndarray,
    kmeans_labels: np.ndarray,
    kmeans_centers_2d: np.ndarray,
    kmedoids_labels: np.ndarray,
    medoid_centers_2d: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    plots = [
        (y_true, "Ground Truth Species"),
        (kmeans_labels, "K-Means on Iris"),
        (kmedoids_labels, "K-Medoids on Iris"),
    ]

    for ax, (labels, title) in zip(axes, plots):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=45, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.3)

    axes[1].scatter(
        kmeans_centers_2d[:, 0],
        kmeans_centers_2d[:, 1],
        marker="X",
        c="black",
        s=180,
        label="Centroid",
    )
    axes[1].legend()

    axes[2].scatter(
        medoid_centers_2d[:, 0],
        medoid_centers_2d[:, 1],
        marker="*",
        c="black",
        s=220,
        label="Medoid",
    )
    axes[2].legend()
    axes[0].legend(*scatter.legend_elements(), title="Class")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_kmedoids_iris_comparison.png", dpi=150)
    plt.close()


def run_kmedoids_iris_demo() -> None:
    print_section("1. K-MEDOIDS: IRIS DATASET")

    iris = load_iris()
    X_scaled = StandardScaler().fit_transform(iris.data)
    y_true = iris.target

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
    labels_kmedoids = kmedoids.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    kmeans_centers_2d = pca.transform(kmeans.cluster_centers_)
    medoid_centers_2d = pca.transform(kmedoids.cluster_centers_)

    print("Dataset: Iris (scikit-learn)")
    print("Target asli dipakai hanya untuk evaluasi ARI, bukan untuk training.")
    print(f"K-Means inertia      : {kmeans.inertia_:.2f}")
    print(f"K-Medoids inertia    : {kmedoids.inertia_:.2f}")
    print(f"ARI K-Means          : {adjusted_rand_score(y_true, labels_kmeans):.3f}")
    print(f"ARI K-Medoids        : {adjusted_rand_score(y_true, labels_kmedoids):.3f}")
    print("Medoid indices       :", kmedoids.medoid_indices_)

    plot_iris_comparison(
        coords,
        y_true,
        labels_kmeans,
        kmeans_centers_2d,
        labels_kmedoids,
        medoid_centers_2d,
    )


def kernel_kmeans_rbf(
    X: np.ndarray,
    n_clusters: int,
    gamma: float,
    max_iter: int = 100,
    n_init: int = 10,
    random_state: int = 42,
    initial_labels: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = X.shape[0]
    kernel = rbf_kernel(X, gamma=gamma)

    def kernel_distance(current_labels: np.ndarray) -> np.ndarray:
        distances = np.zeros((n_samples, n_clusters))
        for cluster_id in range(n_clusters):
            members = current_labels == cluster_id
            if not np.any(members):
                distances[:, cluster_id] = np.inf
                continue
            cluster_size = members.sum()
            within_cluster = kernel[np.ix_(members, members)].sum()
            distances[:, cluster_id] = (
                np.diag(kernel)
                - 2 * kernel[:, members].sum(axis=1) / cluster_size
                + within_cluster / (cluster_size**2)
            )
        return distances

    rng = np.random.default_rng(random_state)
    initializations = []
    if initial_labels is not None:
        initializations.append(initial_labels.astype(int))
    initializations.extend(rng.integers(0, n_clusters, size=n_samples) for _ in range(n_init))

    best_labels = None
    best_cost = np.inf
    for labels in initializations:
        labels = labels.copy()
        if len(np.unique(labels)) < n_clusters:
            labels[:n_clusters] = np.arange(n_clusters)

        for _ in range(max_iter):
            new_labels = kernel_distance(labels).argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        distances = kernel_distance(labels)
        cost = float(distances[np.arange(n_samples), labels].sum())
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("Kernel K-Means failed to initialize labels.")
    return best_labels


def plot_digits_comparison(
    coords: np.ndarray,
    y_true: np.ndarray,
    labels_kmeans: np.ndarray,
    labels_spectral: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    plots = [
        (y_true, "Ground Truth Digits"),
        (labels_kmeans, "K-Means on Digits"),
        (labels_spectral, "SpectralClustering RBF"),
    ]

    for ax, (labels, title) in zip(axes, plots):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.25)

    axes[0].legend(*scatter.legend_elements(), title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_spectral_digits_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_spectral_digits_demo() -> None:
    print_section("2. KERNEL/SPECTRAL IDEA: DIGITS DATASET")

    digits = load_digits()
    X_scaled = StandardScaler().fit_transform(digits.data)
    y_true = digits.target

    kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=10,
        affinity="nearest_neighbors",
        n_neighbors=12,
        assign_labels="kmeans",
        random_state=42,
    )
    labels_spectral = spectral.fit_predict(X_scaled)

    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    print("Dataset: Digits (scikit-learn, 8x8 handwritten digits)")
    print(f"ARI K-Means        : {adjusted_rand_score(y_true, labels_kmeans):.3f}")
    print(f"ARI Spectral       : {adjusted_rand_score(y_true, labels_spectral):.3f}")
    print("Catatan: Spectral memakai similarity graph, konsepnya dekat dengan kernel-based clustering.")

    plot_digits_comparison(coords, y_true, labels_kmeans, labels_spectral)


def plot_circles_comparison(
    X: np.ndarray,
    y_true: np.ndarray,
    labels_kmeans: np.ndarray,
    labels_kernel: np.ndarray,
    labels_spectral: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=True)
    plots = [
        (y_true, "Ground Truth"),
        (labels_kmeans, "K-Means"),
        (labels_kernel, "Kernel K-Means mini"),
        (labels_spectral, "Spectral RBF"),
    ]

    for ax, (labels, title) in zip(axes, plots):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=35, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_kernel_kmeans_circles.png", dpi=150)
    plt.close()


def run_circles_demo() -> None:
    print_section("3. CLASSIC NON-LINEAR BENCHMARK: NOISY CIRCLES")

    X, y_true = make_circles(n_samples=450, factor=0.45, noise=0.06, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    gamma = 8.0

    labels_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_scaled)
    labels_spectral = SpectralClustering(
        n_clusters=2,
        affinity="rbf",
        gamma=gamma,
        random_state=42,
    ).fit_predict(X_scaled)
    labels_kernel = kernel_kmeans_rbf(
        X_scaled,
        n_clusters=2,
        gamma=gamma,
        n_init=10,
        random_state=42,
        initial_labels=labels_spectral,
    )

    print("Dataset: Noisy circles (classic non-linear clustering benchmark)")
    print(f"ARI K-Means          : {adjusted_rand_score(y_true, labels_kmeans):.3f}")
    print(f"ARI Kernel K-Means   : {adjusted_rand_score(y_true, labels_kernel):.3f}")
    print(f"ARI Spectral RBF     : {adjusted_rand_score(y_true, labels_spectral):.3f}")

    plot_circles_comparison(X_scaled, y_true, labels_kmeans, labels_kernel, labels_spectral)

    kernel = rbf_kernel(X_scaled, gamma=gamma)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(kernel, cmap="viridis", aspect="auto")
    ax.set_title(f"RBF Kernel Similarity Matrix (gamma={gamma})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    plt.colorbar(image, ax=ax, label="Similarity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_rbf_kernel_similarity_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    run_kmedoids_iris_demo()
    run_spectral_digits_demo()
    run_circles_demo()
    print_section("OUTPUT")
    print(f"Figures tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
