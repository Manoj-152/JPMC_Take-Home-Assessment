cd Part2_Segmentation

echo "Running data preprocessing..."
python3 data_preprocessing.py
echo ""

echo "Running PCA dimensionality reduction to retain 70% variance..."
python3 pca_dimensionality_reduction.py
echo ""

echo "Running Elbow method to determine optimal number of clusters for KMeans..."
python3 KMeans_elbow_plot_analysis.py
echo ""

echo "Running KMeans clustering and cluster profiling..."
python3 profiling_KMeans_clusters.py
echo ""