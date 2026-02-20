##############################################################################################################
# Script: pca_dimensionality_reduction.py
# This script performs feature augmentation and PCA dimensionality reduction for the segmentation task.
##############################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def feature_augment_for_segmentation(df_full):
    # 1. Citizenship Mapping (Changing to Binary)
    df_full['citizenship_mapped'] = df_full['citizenship'].apply(
        lambda x: 0 if x == 'Foreign born- Not a citizen of U S ' else 1
    )

    # 2. Family Members Under 18 Mapping (reducing dimensionality for better segmentation)
    def map_family_categorical(val):
        if val == 'Not in universe':
            return 'Not in universe'
        elif val == 'Neither parent present':
            return 'parent not present'
        else:
            # Covers: Both parents, Mother only, Father only
            return 'parent present'

    df_full['family_status_cat'] = df_full['family members under 18'].apply(map_family_categorical)
    df_full = df_full.drop(columns=['citizenship', 'family members under 18'])
    return df_full


def pca_dimensionality_reduction():
    # Loading the data (train + test)
    train_df = pd.read_csv('../processed_datas/census_train.csv')
    test_df = pd.read_csv('../processed_datas/census_test.csv')
    df_full = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    df_full = feature_augment_for_segmentation(df_full)

    # Loading the top 25 features (found in previous part)
    df_top_25 = df_full.drop(columns=['label'])

    # We are only using features with <=5 unique categories (for categorical) and all numeric features.
    # For better representation and interpretability purposes
    numeric_features = df_top_25.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in df_top_25.select_dtypes(include=['object','str']).columns 
                            if df_top_25[col].nunique() <= 5]

    X= pd.get_dummies(df_top_25[numeric_features + categorical_features], drop_first=True)
    print(len(X.columns), "features available for PCA.")

    # Scaling is necessary for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fitting PCA without limiting components first
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    # Calculating Cumulative Explained Variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Finding the exact number of components for >=70% variance
    n_components_70 = np.argmax(cumulative_variance >= 0.70) + 1

    print(f"PCA Components required for 70% variance: {n_components_70}")
    return pca_full, X_scaled, n_components_70, df_full, X


if __name__ == "__main__":
    pca_full, X_scaled, n_components_70, _, _ = pca_dimensionality_reduction()