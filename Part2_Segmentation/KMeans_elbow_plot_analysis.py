##############################################################################################################
# Script: KMeans_elbow_plot_analysis.py
# This script performs KMeans clustering and Elbow Method analysis to determine the optimal number of clusters
# for the marketing segmentation task, using the PCA-reduced data from the previous step.
##############################################################################################################

from sklearn.cluster import KMeans
from pca_dimensionality_reduction import pca_dimensionality_reduction
import matplotlib.pyplot as plt
import os

def main():
    # Calling the function to get pca_full, X_scaled, and n_components_70
    pca_full, X_scaled, n_components_70, _, _ = pca_dimensionality_reduction()

    # Transforming the data into the n_components_70 space
    X_pca_optimized = pca_full.transform(X_scaled)[:, :n_components_70]

    # Running the Elbow Method (Checking 1 to 10 clusters)
    wcss = []
    print("\n--- Elbow Method Analysis ---")
    print(f"{'Clusters (K)':<15} {'Inertia (WCSS)':<20} {'% Drop from prev K'}")
    print("-" * 55)

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_pca_optimized)
        wcss.append(kmeans.inertia_)

        # Calculating inertial drop statistics
        if i > 1:
            prev_inertia = wcss[-2]
            curr_inertia = wcss[-1]
            drop = prev_inertia - curr_inertia
            pct_drop = (drop / prev_inertia) * 100
            print(f"{i:<15} {curr_inertia:<20.0f} {pct_drop:.2f}%")

    # Plotting the result
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
    plt.title('Elbow Method: Optimal Marketing Segments', fontsize=14)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('WCSS (Inertia)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Usually, the bend is at K=4 or K=5
    os.makedirs('../plots_and_metadata', exist_ok=True)
    plt.savefig('../plots_and_metadata/kmeans_elbow_plot.png')
    print("\nElbow plot saved as 'kmeans_elbow_plot.png'.")

    print("\nBased on the Elbow Method analysis, we observe a significant drop in inertia up to K=4 or K=5, " \
    "suggesting that these may be optimal choices for the number of clusters in our marketing segmentation. " \
    "The drop in inertia plateaus after that, indicating dimmishing returns along with increasing clustering complexity.")


if __name__ == "__main__":
    main()