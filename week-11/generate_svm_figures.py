from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_decision_boundary(ax, model, X, y, title: str) -> None:
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=["#DBEAFE", "#FED7AA"], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#64748B", "#1D4ED8", "#64748B"], linestyles=["--", "-", "--"], linewidths=[1.5, 2.2, 1.5])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=46, edgecolor="white", linewidth=0.8)
    if hasattr(model, "support_vectors_"):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=140, facecolors="none", edgecolors="#111827", linewidth=1.4, label="support vectors")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def figure_max_margin() -> None:
    X, y = make_blobs(n_samples=60, centers=[[-2, -1], [2, 1]], cluster_std=0.75, random_state=7)
    y = np.where(y == 0, -1, 1)
    model = SVC(kernel="linear", C=100).fit(X, y)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    plot_decision_boundary(ax, model, X, y, "Maximum-margin hyperplane")
    ax.text(0.02, 0.02, "solid line = hyperplane\ndashed lines = margin", transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92})
    savefig("01_svm_max_margin_hyperplane.png")


def figure_c_effect() -> None:
    X, y = make_classification(
        n_samples=140,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=0.85,
        flip_y=0.08,
        random_state=12,
    )
    models = [SVC(kernel="linear", C=c).fit(X, y) for c in [0.1, 100]]
    titles = ["C kecil: margin lebih toleran", "C besar: boundary lebih ketat"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for ax, model, title in zip(axes, models, titles):
        plot_decision_boundary(ax, model, X, y, title)
    savefig("02_svc_c_effect.png")


def figure_gamma_effect() -> None:
    X, y = make_circles(n_samples=220, factor=0.42, noise=0.13, random_state=10)
    gammas = [0.2, 10]
    models = [SVC(kernel="rbf", C=10, gamma=g).fit(X, y) for g in gammas]
    titles = ["gamma kecil: boundary smooth", "gamma besar: boundary lokal"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for ax, model, title in zip(axes, models, titles):
        x_min, x_max = X[:, 0].min() - 0.35, X[:, 0].max() + 0.35
        y_min, y_max = X[:, 1].min() - 0.35, X[:, 1].max() + 0.35
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#DBEAFE", "#FED7AA"], alpha=0.7)
        ax.contour(xx, yy, model.decision_function(grid).reshape(xx.shape), levels=[0], colors="#1D4ED8", linewidths=2.0)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=35, edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    savefig("03_rbf_gamma_effect.png")


def figure_grid_search_heatmap() -> None:
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        class_sep=1.0,
        random_state=42,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    c_values = [2**i for i in range(-3, 8, 2)]
    gamma_values = [2**i for i in range(-9, 4, 2)]
    search = GridSearchCV(
        pipe,
        {"svc__C": c_values, "svc__gamma": gamma_values},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X, y)
    scores = search.cv_results_["mean_test_score"].reshape(len(c_values), len(gamma_values))

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    im = ax.imshow(scores, cmap="Blues", origin="lower", aspect="auto")
    ax.set_xticks(range(len(gamma_values)))
    ax.set_xticklabels([f"2^{i}" for i in range(-9, 4, 2)], rotation=45)
    ax.set_yticks(range(len(c_values)))
    ax.set_yticklabels([f"2^{i}" for i in range(-3, 8, 2)])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("Grid Search CV Score untuk RBF SVC", fontsize=12, fontweight="bold")
    for i in range(len(c_values)):
        for j in range(len(gamma_values)):
            ax.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0F172A")
    best_i, best_j = np.unravel_index(np.argmax(scores), scores.shape)
    ax.scatter([best_j], [best_i], s=220, facecolors="none", edgecolors="#F97316", linewidths=2.5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean CV accuracy")
    savefig("04_grid_search_heatmap.png")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    })
    figure_max_margin()
    figure_c_effect()
    figure_gamma_effect()
    figure_grid_search_heatmap()


if __name__ == "__main__":
    main()
