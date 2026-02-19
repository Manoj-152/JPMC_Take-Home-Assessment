##############################################################################################################
# Script: fitting_best_model_on_all_features.py
# This script fits the champion Random Forest model using all 25 engineered features and evaluates its performance
# on the hold-out test set, providing a comparision benchmark.
##############################################################################################################

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # Loading Data
    train_df = pd.read_csv('../processed_datas/census_train.csv')
    test_df = pd.read_csv('../processed_datas/census_test.csv')

    # Using all available 25 engineered features
    all_25_features = [col for col in train_df.columns if col != 'label']

    # Preprocessing
    X_train_25 = pd.get_dummies(train_df[all_25_features])
    X_test_25 = pd.get_dummies(test_df[all_25_features])
    X_train_25, X_test_25 = X_train_25.align(X_test_25, join='left', axis=1, fill_value=0)

    y_train = train_df['label']
    y_test = test_df['label']

    # Fitting Champion RF with 25 Features
    rf_25 = RandomForestClassifier(
        class_weight='balanced', 
        max_features=0.5, 
        min_samples_leaf=2, 
        n_estimators=200, 
        n_jobs=-1, 
        random_state=RANDOM_SEED
    )

    print(f"Training Champion RF on initial {len(all_25_features)} features...")
    rf_25.fit(X_train_25, y_train)

    # Evaluation
    y_pred_25 = rf_25.predict(X_test_25)
    f1_25 = f1_score(y_test, y_pred_25)

    print("\n" + "="*40)
    print("RESULTS FOR 25-FEATURE MODEL")
    print("="*40)
    print(f"Test F1-Score: {f1_25:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_25))


if __name__ == "__main__":
    main()