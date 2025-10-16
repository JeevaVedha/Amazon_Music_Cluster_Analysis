import pandas as pd
import streamlit as st

# --- Load Data ---
def load_data(file_path):
    return pd.read_csv(file_path)

# --- Basic Checks ---
def check_duplicates(df):
    return df.duplicated().sum()

def check_missing_values(df):
    return df.isnull().sum()

def check_missing_percentage(df):
    return (df.isnull().sum() / len(df)) * 100

# --- Drop Unnecessary Columns ---
def drop_columns(df, columns):
    return df.drop(columns=columns, errors='ignore')

# --- Streamlit App ---
def main():
    st.title("🎵 Spotify Single Genre Artists Data Overview")

    # Load data
    df = load_data(r'C:/Users/Jeeva/Documents/Amazon Music/Amazon_Music_Cluster_Analysis/Data/single_genre_artists.csv')
    st.write("Data loaded successfully!")

    # Display head
    st.subheader("📋 Preview of DataFrame")
    st.dataframe(df.head())

    # Info section
    st.subheader("ℹ️ DataFrame Info")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

    # Descriptive stats
    st.subheader("📊 Statistical Summary")
    st.write(df.describe())

    # Shape
    st.write(f"Shape of DataFrame: {df.shape}")

    # Duplicate rows
    st.write(f"Number of duplicated rows: {check_duplicates(df)}")

    # Missing values
    st.subheader("🧩 Missing Values")
    st.write(check_missing_values(df))
    st.write("Percentage of missing values:")
    st.write(check_missing_percentage(df))

    # Drop unnecessary columns
    drop_list = ['id_songs', 'name_song', 'id_artists', 'genres', 'name_artists']
    df = drop_columns(df, drop_list)

    # Convert date to year
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df.drop(columns=['release_date'], inplace=True)

    # Convert datatypes safely
    df = df.astype({
        'explicit': 'int',
        'popularity_songs': 'int',
        'duration_ms': 'int',
        'key': 'int',
        'mode': 'int',
        'time_signature': 'int',
        'followers': 'int',
        'popularity_artists': 'int',
        'danceability': 'float',
        'energy': 'float',
        'loudness': 'float',
        'speechiness': 'float',
        'acousticness': 'float',
        'instrumentalness': 'float',
        'liveness': 'float',
        'valence': 'float',
        'tempo': 'float'
    })

    # Display cleaned DataFrame
    st.subheader("✅ Cleaned DataFrame")
    st.dataframe(df.head())

    st.success("Data preprocessing completed successfully!")

# --- Run Streamlit App ---
if __name__ == "__main__":
    main()
