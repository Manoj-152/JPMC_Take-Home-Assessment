import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pca_dimensionality_reduction import pca_dimensionality_reduction
import os


def main():
    pca_full, X_scaled, n_components_70, df_full, _ = pca_dimensionality_reduction()

    X_pca_optimized = pca_full.transform(X_scaled)[:, :9]
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    df_full['Cluster'] = kmeans.fit_predict(X_pca_optimized) # Using your optimal components

    ####### The Persona Decoder ##########
    # This aggregates the ORIGINAL features to explain who these people are
    persona_summary = df_full.groupby('Cluster').agg({
        'label': 'mean',            # High Earner
        'age': 'mean',
        'education_num': 'mean',
        'total_investment': 'mean', # Capital gain/loss feature
        'wage per hour': 'mean',
        'weeks worked in year': 'mean',
        'citizenship_mapped': 'mean',
        'family_status_cat': lambda x: x.mode()[0] # Most common family status
    }).round(2)

    print("\n--- FINAL MARKETING PERSONAS (FOR 4 CLUSTERS) ---")
    print(persona_summary)

    # Checking the size of each cluster to ensure they aren't tiny
    print("\n--- Cluster Sizes ---")
    print(df_full['Cluster'].value_counts())


    # Preparing data for bar chart
    # We normalize the data to 0-1 scale just for this plot so we can compare "Age" with "Weeks Worked"
    from sklearn.preprocessing import MinMaxScaler

    # Selecting features to visualize
    features_to_plot = ['age', 'education_num', 'total_investment', 'weeks worked in year', 'label', 'wage per hour']
    df_means = df_full.groupby('Cluster')[features_to_plot].mean()

    # Scaling them so they fit on one chart (0 to 1)
    scaler_plot = MinMaxScaler()
    df_means_scaled = pd.DataFrame(scaler_plot.fit_transform(df_means), 
                                columns=df_means.columns, 
                                index=df_means.index)

    # Plotting
    df_means_scaled.plot(kind='bar', figsize=(12, 6), colormap='Set2', width=0.8)
    plt.title('Persona Summarization: Normalized Feature Values by Cluster', fontsize=15)
    plt.ylabel('Relative Strength (0 to 1)', fontsize=12)
    plt.xlabel('Cluster (Persona)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    os.makedirs('../plots_and_metadata', exist_ok=True)
    plt.savefig('../plots_and_metadata/cluster_persona_profiles.png')
    print("\nCluster persona profiles saved as 'cluster_persona_profiles.png'.")


if __name__ == "__main__":
    main()