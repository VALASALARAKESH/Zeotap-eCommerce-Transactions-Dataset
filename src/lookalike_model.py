import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, MeanShift, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from fpdf import FPDF

# Load Data
customers_path = "../data/Customers.csv"
transactions_path = "../data/Transactions.csv"

customers = pd.read_csv(customers_path)
transactions = pd.read_csv(transactions_path)

# Preprocessing Customers Data
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['SignupYear'] = customers['SignupDate'].dt.year

# Aggregating Transactions Data
transaction_agg = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count',
    'Quantity': 'sum'
}).rename(columns={
    'TransactionID': 'TransactionCount',
    'Quantity': 'TotalQuantity'
}).reset_index()

# Merge Data
merged_data = pd.merge(customers, transaction_agg, on='CustomerID', how='left')
merged_data.fillna(0, inplace=True)

# Feature Selection
features = merged_data[['Region', 'SignupYear', 'TransactionCount', 'TotalValue', 'TotalQuantity']]
features = pd.get_dummies(features, columns=['Region'], drop_first=True)

# Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Clustering Algorithms Dictionary
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=4, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Spectral': SpectralClustering(n_clusters=4, random_state=42),
    'MeanShift': MeanShift(),
    'Birch': Birch(n_clusters=4),
    'OPTICS': OPTICS(min_samples=5),
    'GaussianMixture': GaussianMixture(n_components=4, random_state=42)
}

# Function to Apply Clustering Algorithm and Compute DB Index
def apply_clustering_algorithm(algorithm, model, features):
    model.fit(features)
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(features)
    if len(set(labels)) > 1:  # Ensure there is more than one cluster
        db_index = davies_bouldin_score(features, labels)
    else:
        db_index = float('inf')  # Assign a high value if only one cluster
    return labels, db_index

# Prepare PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Clustering Report: eCommerce Transactions Dataset", ln=True, align="C")
pdf.ln(10)

# Store visualizations and DB Indexes for each algorithm
plots = []
db_indexes = []

# Apply each clustering algorithm and save results
for algorithm, model in clustering_algorithms.items():
    labels, db_index = apply_clustering_algorithm(algorithm, model, scaled_features)
    merged_data[f'{algorithm}_Cluster'] = labels
    db_indexes.append((algorithm, db_index))

    # Cluster Visualization
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=scaled_features[:, 0], y=scaled_features[:, 1],
        hue=labels, palette='Set2'
    )
    plt.title(f"{algorithm} Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plot_filename = f"../output/{algorithm}_cluster_visualization.png"
    plt.savefig(plot_filename)
    plt.close()
    plots.append((f"{algorithm} Cluster Visualization", plot_filename))

    # Pairplot for Pairwise Feature Relationships
    pairplot_df = pd.DataFrame(scaled_features, columns=features.columns)
    pairplot_df['Cluster'] = labels
    pairplot_filename = f"../output/{algorithm}_pairplot.png"
    sns.pairplot(pairplot_df, hue='Cluster')
    plt.savefig(pairplot_filename)
    plt.close()
    plots.append((f"{algorithm} Pairplot", pairplot_filename))

    # Silhouette Plot
    if len(set(labels)) > 1:  # Ensure there is more than one cluster
        silhouette_avg = silhouette_score(scaled_features, labels)
        plt.figure(figsize=(8, 5))
        plt.title(f"Silhouette Plot for {algorithm} - Score: {silhouette_avg:.3f}")
        plt.barh(range(len(labels)), silhouette_score(scaled_features, labels), color='skyblue')
        silhouette_plot_filename = f"../output/{algorithm}_silhouette_plot.png"
        plt.savefig(silhouette_plot_filename)
        plt.close()
        plots.append((f"{algorithm} Silhouette Plot", silhouette_plot_filename))

    # Dendrogram for Agglomerative Clustering (if applicable)
    if algorithm == 'Agglomerative':
        Z = linkage(scaled_features, 'ward')
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title(f"Dendrogram for {algorithm}")
        dendrogram_plot_filename = f"../output/{algorithm}_dendrogram.png"
        plt.savefig(dendrogram_plot_filename)
        plt.close()
        plots.append((f"{algorithm} Dendrogram", dendrogram_plot_filename))

# Add DB Indexes to PDF
pdf.set_font("Arial", size=10)
for algorithm, db_index in db_indexes:
    pdf.cell(200, 10, txt=f"{algorithm} - Davies-Bouldin Index: {db_index:.4f}", ln=True)

# Add visualizations to PDF
for title, plot in plots:
    pdf.add_page()
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.image(plot, x=10, y=30, w=190)

# Save PDF report
pdf_output_path = "../output/Rakesh_Valasala_Clustering.pdf"
pdf.output(pdf_output_path)

# Save merged data with cluster labels
output_path = "../output/Rakesh_Valasala_Clustering.csv"
merged_data.to_csv(output_path, index=False)

print(f"Clustering report saved to {pdf_output_path}")
print(f"Clustered data saved to {output_path}")