import pandas as pd
import streamlit as st  

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

import matplotlib.pyplot as plt

# Data Preprocessing
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

st.write("Standardization and Train-Test Split complete.")
plt.figure(figsize=(10, 6))
plt.plot(X_train[:, 0], X_train[:, 1], 'o', markersize=5)
plt.title("Standardized Features Scatter Plot")
st.pyplot(plt)

st.write("Data Preprocessing complete.")

plt.figure(figsize=(10, 6)) 
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
st.pyplot(plt)
optimal_k = 3  # This should be determined from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_train)

train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)    

st.write("K-Means Clustering complete.")
st.write(f"Optinal number of clusters (K): {optimal_k}")


plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title("K-Means Clustering Results")
st.pyplot(plt)
print("ML Model training and evaluation complete.")