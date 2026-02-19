cd Part1_Classifier

echo "Running data preprocessing..."
python3 data_preprocessing.py
echo ""

echo "Running NMI-based feature selection..."
python3 NMI_based_feature_selection.py
echo ""

echo "Running model training and hyperparameter tuning..."
python3 model_hyperparam_tuning.py
echo ""

echo "Running model fitting on all features to set comparison with best model from feature selection..."
python3 fitting_best_model_on_all_features.py
echo ""