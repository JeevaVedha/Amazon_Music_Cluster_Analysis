## 🎵 Amazon Music Cluster Analysis Dashboard

This project builds an interactive Streamlit dashboard to perform exploratory data analysis (EDA) and K-Means clustering on Amazon Music data.
It visualizes PCA transformation, feature correlations, and clustering performance using the Elbow Method and Silhouette Score.

## 📘 Overview

The goal of this project is to:

Analyze and clean the Amazon Music dataset.

Perform feature scaling and dimensionality reduction (PCA).

Apply K-Means clustering to group similar music tracks.

Evaluate cluster performance using Elbow and Silhouette methods.

Visualize results interactively in Streamlit.

## 🧩 Key Features

✅ Data Preprocessing – Missing values, duplicates, and unnecessary columns are handled.

📊 Exploratory Data Analysis – Statistical summaries, heatmaps, and PCA visualizations.

🧠 Machine Learning Pipeline – K-Means clustering and PCA-based visualization.

🔍 Cluster Evaluation – Uses both Elbow and Silhouette methods to determine the best number of clusters.

🪶 Cluster Profiling – Automatically generates descriptive labels for each cluster.

💾 Data Export – Download the final clustered dataset as a CSV file.

## 🧠 Technologies Used
| Category                 | Tools / Libraries   |
| ------------------------ | ------------------- |
| Language                 | Python              |
| Dashboard Framework      | Streamlit           |
| Data Manipulation        | Pandas, NumPy       |
| Visualization            | Matplotlib, Seaborn |
| Machine Learning         | Scikit-learn        |
| Dimensionality Reduction | PCA, t-SNE          |


## 📂 Project Structure
Amazon_Music_Cluster_Analysis/
│
├── Data/
│   └── single_genre_artists.csv
│
├── app.py                # Streamlit dashboard code
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies

## ⚙️ Installation & Setup
### 1. Clone the repository

git clone https://github.com/JeevaVedha/Amazon_Music_Cluster_Analysis.git
cd Amazon_Music_Cluster_Analysis

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the Streamlit app

streamlit run app.py

### 4. Load the dataset

Make sure the file single_genre_artists.csv is available in the Data/ folder.

## 📈 Workflow Summary

Data Loading

Loads the dataset and checks for missing or duplicate rows.

Data Cleaning

Drops unnecessary columns (id_songs, id_artists, etc.).

Checks missing and null values.

Feature Scaling

Applies StandardScaler to normalize numerical data.

Dimensionality Reduction

Uses PCA to reduce feature space and visualize high-dimensional data.

Clustering

Applies K-Means clustering to segment the dataset.

Determines optimal number of clusters using Elbow and Silhouette Score.

Visualization

Displays:

Explained variance ratio (scree plot)

Correlation heatmap

PCA 2D plot

Elbow curve

Silhouette score graph

Cluster Profiling

Summarizes each cluster’s key characteristics (e.g., Party Tracks, Chill Acoustic).

Data Export

Provides option to download final clustered data as CSV.

## 📊 Example Outputs

Scree Plot showing explained variance by PCA components.

Heatmap highlighting feature correlations.

2D PCA Scatter Plot for visualizing music clusters.

Cluster Summary Table showing mean values of features by cluster.

## 🧮 Example Cluster Profiles
Cluster	Description
0	💃 Party Tracks — High Danceability & Energy
1	🎸 Chill Acoustic — Calm and Relaxed
2	🎤 Vocal / Rap-heavy Tracks
3	🎹 Instrumental / Ambient Music