##############################################################################################################
# Script: model_hyperparam_tuning.py
# This script performs hyperparameter tuning for Random Forest, XGBoost, and LightGBM classifiers using 
# GridSearchCV and evaluates their performance on the hold-out test set.
##############################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import time
import joblib
import os


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # Loading Top 14 features
    with open('../processed_datas/top_14_features.columns', 'r') as f:
        top_14 = [line.strip() for line in f]

    train_df = pd.read_csv('../processed_datas/census_train.csv')
    test_df = pd.read_csv('../processed_datas/census_test.csv')
    # Preprocessing: One-hot encode and align
    X_train = pd.get_dummies(train_df[top_14])
    X_test = pd.get_dummies(test_df[top_14])
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    y_train = train_df['label']
    y_test = test_df['label']

    # Imbalance Weight creation for XGBoost and LightGBM
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    # Defining the Grids (48 + 48 + 54 Combinations)
    model_specs = [
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
            'grid': {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, 25, None],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 0.5]
            }
        },
        {
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=RANDOM_SEED, eval_metric='logloss', n_jobs=-1),
            'grid': {
                'n_estimators': [200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 9, 12],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.7, 0.8],
            }
        },
        {
            'name': 'LightGBM',
            'model': lgb.LGBMClassifier(scale_pos_weight=pos_weight, random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1),
            'grid': {
                'n_estimators': [200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 64, 128],
                'reg_alpha': [0, 0.1, 0.5],
                'boosting_type': ['gbdt', 'dart']
            }
        }
    ]

    # Execution Loop
    best_overall_f1 = 0
    champion_model = None
    tournament_results = []

    print(f"Starting Tournament with {len(model_specs)} model specifications...")

    for spec in model_specs:
        start_time = time.time()
        print(f"\nRunning Grid Search for {spec['name']}...")
        
        grid_search = GridSearchCV(spec['model'], spec['grid'], cv=cv, scoring='f1', n_jobs=1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Evaluating on hold-out Test Set
        best_mod = grid_search.best_estimator_
        y_pred = best_mod.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        
        elapsed = time.time() - start_time
        print(f"Done! Time: {elapsed:.1f}s | Best Test F1: {test_f1:.4f}")
        
        tournament_results.append({
            'Model': spec['name'],
            'Best F1': test_f1,
            'Params': grid_search.best_params_
        })
        
        if test_f1 > best_overall_f1:
            best_overall_f1 = test_f1
            champion_model = best_mod

    # Final Reporting
    print("\n" + "="*40)
    print("FINAL RANKINGS")
    print("="*40)
    ranking_df = pd.DataFrame(tournament_results).sort_values('Best F1', ascending=False)
    print(ranking_df[['Model', 'Best F1']].to_string(index=False))

    os.makedirs('../plots_and_metadata', exist_ok=True)
    ranking_df.to_csv('../plots_and_metadata/model_tournament_ranking.csv', index=False)

    print("\nCHAMPION MODEL DETAILS:")
    print(champion_model)
    y_final_pred = champion_model.predict(X_test)
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_final_pred))

    # Saving the champion model
    joblib.dump(champion_model, '../champion_income_classifier_model.pkl')


if __name__ == "__main__":
    main()