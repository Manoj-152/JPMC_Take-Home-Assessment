# Income Classification & Population Segmentation for Targeted Marketing

[![Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/Manoj-152/JPMC_Take-Home-Assessment)

This repository contains a dual-pronged machine learning framework developed for targeted marketing optimization. The project is divided into two core analytical components:
1. **Part 1 (Classification):** A supervised machine learning pipeline (Random Forest) designed to predict high-income individuals (>$50k) using census data, optimized via Normalized Mutual Information (NMI) feature selection.
2. **Part 2 (Segmentation):** An unsupervised learning approach utilizing Principal Component Analysis (PCA) and K-Means clustering to discover hidden socio-economic personas for strategic product mapping.

## 📂 Repository Structure

```text
├── Part1_Classifier/
│   ├── data_preprocessing.py
│   ├── NMI_based_feature_selection.py
│   ├── model_hyperparam_tuning.py
│   ├── fitting_best_model_on_all_features.py
├── main_part1_classifier.ipynb     # Interactive notebook for Part 1
├── main_part1_classifier.sh        # Executable bash script for Part 1
├── Part2_Segmentation/
│   ├── data_preprocessing.py
│   ├── pca_dimensionality_reduction.py
│   ├── KMeans_elbow_plot_analysis.py
│   ├── profiling_KMeans_clusters.py
├── main_part2_segmentation.sh      # Executable bash script for Part 2
├── main_part2_segmentation.ipynb   # Interactive notebook for Part 2
├── census-bureau.data             # Raw census data file (ensure this is downloaded and placed here)
├── census-bureau.columns          # Column names for the census data (ensure this is downloaded and placed here)                           
├── README.md
├── requirements.txt
└──JPMC-Take_Home_Assessment-Report.pdf # Project report detailing methodologies, results, and insights
```

## ⚙️ Prerequisites & Setup

Ensure you have **Python 3.8+** installed. To replicate the environment and install all necessary dependencies (like `pandas`, `scikit-learn`, `numpy`, `matplotlib`, and `seaborn`), run:

```bash
pip install -r requirements.txt
```
*(Note: Please ensure the raw census data files are downloaded and placed in the appropriate working directories before execution).*

## 🚀 Execution Instructions

You can evaluate the codebase using either the interactive Jupyter Notebooks or directly via the command line.

### Option A: Interactive Execution (Jupyter Notebooks)
For a visual, step-by-step walkthrough of the code, data transformations, and model evaluation metrics:
1. Launch Jupyter Notebook or Jupyter Lab from your terminal.
2. Open `main_part1_classifier.ipynb` to execute the supervised classification pipeline.
3. Open `main_part2_segmentation.ipynb` to execute the unsupervised clustering pipeline.
4. Run all cells sequentially.

### Option B: Terminal Execution (Scripts)
If you prefer to run the raw Python scripts directly from your terminal, you can execute them sequentially. 

**Running Part 1: Classification**
```bash
bash main_part1_classifier.sh
```

**Running Part 2: Segmentation**
```bash
bash main_part2_segmentation.sh
```

## 👤 Author
**Manoj Srinivasan** *Data Science / Machine Learning*