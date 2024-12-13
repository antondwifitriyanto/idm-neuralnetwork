import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load data
def load_data():
    file_path = 'DataTrainingKabSerang.xlsx'  # Adjust path as needed
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    return df

data = load_data()

# Sidebar for filters
st.sidebar.header("Filter Data")
year = st.sidebar.selectbox("Pilih Tahun", options=[2021, 2022, 2023, 2024])

# Filter data by year
iks_col = f"IKS {year}"
ike_col = f"IKE {year}"
ikl_col = f"IKL {year}"
idm_col = f"NILAI IDM {year}"
status_col = f"STATUS IDM {year}"

filtered_data = data[["NAMA DESA", iks_col, ike_col, ikl_col, idm_col, status_col]].rename(
    columns={
        iks_col: "IKS",
        ike_col: "IKE",
        ikl_col: "IKL",
        idm_col: "NILAI IDM",
        status_col: "STATUS IDM",
    }
)

# Display data
st.title("Indeks Desa Membangun")
st.write(f"Menampilkan data IDM tahun {year}")
st.dataframe(filtered_data)

# Visualize data
st.subheader("Visualisasi Data")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot of IKS vs IKE
ax[0].scatter(filtered_data["IKS"], filtered_data["IKE"], alpha=0.7, c="blue")
ax[0].set_title("IKS vs IKE")
ax[0].set_xlabel("IKS")
ax[0].set_ylabel("IKE")

# Scatter plot of IKL vs NILAI IDM
ax[1].scatter(filtered_data["IKL"], filtered_data["NILAI IDM"], alpha=0.7, c="green")
ax[1].set_title("IKL vs NILAI IDM")
ax[1].set_xlabel("IKL")
ax[1].set_ylabel("NILAI IDM")

st.pyplot(fig)

# Clustering with K-Means
st.subheader("Clustering Desa")
number_of_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)

# Prepare data for clustering
scaler = MinMaxScaler()
clustering_data = filtered_data[["IKS", "IKE", "IKL"]]
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Apply K-Means
kmeans = KMeans(n_clusters=number_of_clusters, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)
filtered_data["Cluster"] = clusters

# Show cluster results
st.write("Data dengan Cluster:")
st.dataframe(filtered_data)

# Visualize clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    filtered_data["IKS"],
    filtered_data["IKE"],
    c=filtered_data["Cluster"],
    cmap="viridis",
    alpha=0.7
)
ax.set_title("Clustering Desa Berdasarkan IKS dan IKE")
ax.set_xlabel("IKS")
ax.set_ylabel("IKE")
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
st.pyplot(fig)

st.success("Analisis selesai! Anda dapat mengunduh hasil clustering jika diperlukan.")
