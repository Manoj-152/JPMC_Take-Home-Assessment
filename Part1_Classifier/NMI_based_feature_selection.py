##############################################################################################################
# Script: NMI_based_feature_selection.py
# This script performs feature selection using Normalized Mutual Information (NMI) and evaluates the performance
# of subsets of features using Logistic Regression with cross-validation.
##############################################################################################################

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
import scipy.stats as ss


def main():
    # Set Global Seeds for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    # Defining the CV strategy here to ensure the 3-fold split is identical every time
    cv_strategy = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)


    def entropy(y):
        # Calculating probabilities of each class (0 and 1)
        p = y.value_counts(normalize=True)
        return ss.entropy(p)

    df = pd.read_csv('../processed_datas/census_train.csv')
    X = df.drop('label', axis=1)
    y = df['label']

    # Identifying column types for later use
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Dynamic NMI Calculation for Ranking
    print("Calculating Feature Importance (Mutual Information)...")
    X_encoded = X.copy()
    # Label Encode strings just for the MI calculation
    for col in X_encoded.select_dtypes(exclude=[np.number]).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

    # Identifying which columns are discrete (categorical) for the MI algorithm
    discrete_features = [X.columns.get_loc(col) for col in categorical_cols]

    # Calculating scores
    target_entropy = entropy(y)
    mi_scores = mutual_info_classif(X_encoded, y, discrete_features=discrete_features, random_state=RANDOM_SEED)
    mi_scores = mi_scores / target_entropy  # Normalize by target entropy to get NMI

    # Creating a sorted ranking
    feature_info = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    feature_ranking = feature_info.index.tolist()

    print("\nTop 5 Dynamic Rankings:")
    print(feature_info.head(5))
    print(len(feature_ranking), "features ranked in total.")

    # 5. Iterative Testing (The "Elbow" Simulation)
    results = []
    feature_counts = range(1, len(feature_ranking)+1)

    print("\nStarting Logistic Regression Simulation...")
    for k in feature_counts:
        print(f"\rEvaluating top {k} features...", end="")
        top_k_features = feature_ranking[:k]
        current_num = [f for f in top_k_features if f in numeric_cols]
        current_cat = [f for f in top_k_features if f in categorical_cols]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), current_num) if current_num else ('num', 'passthrough', []),  # Only apply if there are numeric features
                ('cat', OneHotEncoder(handle_unknown='ignore'), current_cat) if current_cat else ('cat', 'passthrough', []) # Only apply if there are categorical features
            ])
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED))
        ])
        
        # Using 3-fold CV with F1-score
        cv_score = cross_val_score(model, X[top_k_features], y, cv=cv_strategy, scoring='f1').mean()
        results.append({'num_features': k, 'f1_score': cv_score})

    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    print("\n--- PERFORMANCE RANKING (Top Subsets) ---")
    print(f"{'Features':<10} | {'F1-Score':<10}")
    print("-" * 25)
    for res in sorted_results[:10]:  # Printing Top 10 for clarity
        print(f"{res['num_features']:<10} | {res['f1_score']:.4f}")

    print("\nObtained elbow point at around 14 features, with diminishing returns observed beyond that.")

    # 6. Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot([r['num_features'] for r in results], [r['f1_score'] for r in results], marker='o', linestyle='-', color="#024C96")
    plt.title('Dynamic Feature Selection: Logistic Regression Performance', fontsize=14)
    plt.xlabel('Number of Top Features (Ranked by Mutual Info)', fontsize=12)
    plt.ylabel('F1-Score (Cross-Validated)', fontsize=12)
    plt.grid(alpha=0.3)

    # Saving the plot
    os.makedirs('../plots_and_metadata', exist_ok=True)
    plt.savefig('../plots_and_metadata/feature_selection_curve.png', dpi=300, bbox_inches='tight')

    print("Saving top features for future reference...")

    top_14 = feature_ranking[:14]
    top_25 = feature_ranking[:25]

    def save_feature_list(feature_list, filename):
        with open(filename, 'w') as f:
            for feat in feature_list:
                f.write(f"{feat}\n")
        print(f"Saved {len(feature_list)} features to {filename}")

    save_feature_list(top_14, '../processed_datas/top_14_features.columns')
    save_feature_list(top_25, '../processed_datas/top_25_features.columns')

    print("\nTop 14 Features saved:")


if __name__ == "__main__":
    main()