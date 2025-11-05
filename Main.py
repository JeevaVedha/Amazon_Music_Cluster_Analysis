import pandas as pd
import streamlit as st
import numpy as np  

df = pd.read_csv(r'C:/Users/Jeeva/Documents/Amazon Music/Amazon_Music_Cluster_Analysis/Data/single_genre_artists.csv')
print("Data loaded successfully!")  
print(df.head())
print("\nDataFrame Info:")


def check_duplicates(dataframe):
    return dataframe.duplicated().sum() > 0
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print(f"\nShape of DataFrame: {df.shape}")
print(f"Number of duplicated rows: {df.duplicated().sum()}")


def check_missing_values(dataframe):
    return dataframe.isnull().sum() > 0
print("\nMissing Values:")
print(df.isnull().sum())    
print("\nPercentage of missing values:")
print((df.isnull().sum() / len(df)) * 100)
def drop_columns(dataframe, columns):
    return dataframe.drop(columns=columns, errors='ignore')
drop_list = ['id_songs', 'name_song', 'id_artists', 'genres', 'name_artists']
df = drop_columns(df, drop_list)
print("\nDataFrame after dropping unnecessary columns:")
print(df.head())
print(df.info())
print("\nDataFrame Info after dropping columns:")
print(df.info())

def main():
    check_duplicates(df)
    check_missing_values(df)
    drop_columns(df, drop_list)
main()

print("\nAnalysis complete.")


#---- ML Model ----#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
# Data Preprocessing
numeric_df = df.select_dtypes(include=['float64', 'int64'])

if not numeric_df.empty:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

st.title("🎵 Amazon Music Cluster Analysis")

# PCA for Dimensionality Reduction
n_components = st.sidebar.slider("Select number of PCA components", 2, min(5, numeric_df.shape[1]), 3)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(
    X_pca,
    columns=[f'Principal Component {i+1}' for i in range(n_components)]
)
st.write("PCA Transformation complete.")

#Explained Variance
explained_var = pca.explained_variance_ratio_

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("📈 Explained Variance Ratio")
    plt.figure(figsize=(7, 5))
    x_vals = list(range(1, len(explained_var) + 1))
    plt.plot(x_vals, explained_var, marker='o', linestyle='--', color='b')

# add labels for each point
    for x, y in zip(x_vals, explained_var):
        plt.annotate(f"{y:.2f}",  # format to 2 decimal places
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, 8),  # offset above the point
                     ha='center',
                     fontsize=8,
                     color='darkred')

        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.grid(True)

    st.pyplot(plt)
with col2:
    st.subheader("🔥 Feature Correlation Heatmap")

# Compute correlation matrix for numeric features
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
)
    ax.set_title("Correlation Heatmap of Features", fontsize=14)
    st.pyplot(fig)

with col3:
    #PCA 2D Visualization
    if n_components >= 2:
        st.subheader("📊 PCA 2D Visualization")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
        x=pca_df['Principal Component 1'],
        y=pca_df['Principal Component 2'],
        s=80, edgecolor='k'
    )
    plt.title('PCA - 2D Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)
st.success("🎯 PCA Visualization Complete!")    


# Elbow Method (SSE vs K)
col1, col2 = st.columns(2)
with col1:
        st.subheader("Elbow Method — Determine Optimal Number of Clusters")

        max_k = st.sidebar.slider("Select maximum k to test", 2, 15, 3)
        sse = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            sse.append(kmeans.inertia_)

        # Plot SSE vs K
        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
        ax_elbow.plot(range(1, max_k + 1), sse, marker='o', linestyle='-', color='b')
        ax_elbow.set_xlabel("Number of Clusters (k)")
        ax_elbow.set_ylabel("SSE (Inertia)")
        ax_elbow.set_title("Elbow Method for Optimal K")
        st.pyplot(fig_elbow)

        st.info("💡 Look for the 'elbow' point — where the decrease in SSE starts to slow down.")
with col2:

        # Silhouette Score Evaluation
        # -------------------------------------------------------------
       st.subheader("Silhouette Score — Evaluate Cluster Quality")

       # Subsample data for scoring
       sample_size = 20000
       if scaled_features.shape[0] > sample_size:
           idx = np.random.choice(scaled_features.shape[0], sample_size, replace=False)
           sample_data = scaled_features[idx]
       else:
           sample_data = scaled_features
       
       # Define range of k (for example check every 2 values only if large)
       range_k = list(range(2, max_k + 1, 2)) if max_k > 20 else list(range(2, max_k + 1))
       silhouette_scores = []
       
       for k in range_k:
           kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10, max_iter=300)
           kmeans.fit(scaled_features)  # full dataset for fitting
           labels = kmeans.labels_
           score = silhouette_score(sample_data, labels[idx])  # only score sample
           silhouette_scores.append(score)
       
       # Plot results
       fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
       ax_sil.plot(range_k, silhouette_scores, marker='o', linestyle='-', color='g')
       ax_sil.set_xlabel("Number of Clusters (k)")
       ax_sil.set_ylabel("Silhouette Score")
       ax_sil.set_title("Silhouette Score for Different K Values")
       st.pyplot(fig_sil)
       
       best_k = range_k[np.argmax(silhouette_scores)]
       st.success(f"Optimal k based on Silhouette Score: **{best_k}**")

# clustering using K-Means clustering

st.subheader("Apply K-Means Clustering")

num_clusters = st.sidebar.slider("Select number of clusters (k) to apply", 2, max_k, best_k)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_

st.success("K-Means clustering applied successfully!")
st.write(df.head())

# -------------------------------------------------------------
#PCA Visualization (Optional)
# -------------------------------------------------------------
st.subheader("PCA Visualization of Clusters")

pca = PCA(n_components=2)
st.write("Transforming data to 2D using PCA for visualization...")
pca_data = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster']

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x='PCA1', y='PCA2',
    hue='Cluster',
    palette='tab10',
    s=80
)
plt.title("K-Means Clusters (PCA Projection)")
st.pyplot(fig)


# -------------------------------------------------------------
# Cluster Summary
# -------------------------------------------------------------
st.subheader("Cluster Summary (Mean Feature Values)")
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
st.dataframe(cluster_summary)


# -------------------------------------------------------------
# Step 6 — Interpret Clusters (Automatic Profiling)
# -------------------------------------------------------------
st.subheader("🎵 Step 6: Interpret Cluster Profiles")

# Normalize cluster means for easier comparison
norm_summary = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())

profiles = []
for i, row in norm_summary.iterrows():
    high_features = row[row > 0.6].index.tolist()
    low_features = row[row < 0.4].index.tolist()

    profile = ""
    if "danceability" in high_features and "energy" in high_features:
        profile = "💃 Party Tracks — High Danceability & Energy"
    elif "acousticness" in high_features and "energy" in low_features:
        profile = "🎸 Chill Acoustic — Calm and Relaxed"
    elif "speechiness" in high_features:
        profile = "🎤 Vocal / Rap-heavy Tracks"
    elif "instrumentalness" in high_features:
        profile = "🎹 Instrumental / Ambient Music"
    else:
        profile = "🎶 Mixed Style Cluster"

    profiles.append(profile)

# Display results
profile_df = pd.DataFrame({
    "Cluster": cluster_summary.index,
    "Profile Description": profiles
})

st.table(profile_df)

# Option to Download Clustered Data
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Clustered Data (CSV)", csv, "kmeans_clusters.csv", "text/csv")

st.success("✅ K-Means Clustering, Elbow Method, and Silhouette Analysis completed successfully!")
