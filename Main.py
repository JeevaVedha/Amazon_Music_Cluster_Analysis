import pandas as pd
import streamlit as st

df = pd.read_csv('Data/single_genre_artists.csv')
# Display basic information about the DataFrame
st.title("DataFrame Overview")
st.write(df.head())
st.write(df.info())
st.write(df.describe())
st.write(df.shape)

# Check for duplicated rows
duplicated_rows = df[df.duplicated()]
st.write(f"Number of duplicated rows: {duplicated_rows.shape[0]}")

# Check for missing values
missing_values = df.isnull().sum()
st.write("Missing values in each column:")
st.write(missing_values)

#Drop unnecessary columns
df = df.drop(columns=['id_songs','name_song','id_artists','release_date','genres','name_artists'])

st.write("DataFrame after dropping unnecessary columns:")
st.write(df.head())


