import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# Load data
def load_data():
    file_path = 'DataTrainingKabSerang.xlsx'  # Adjust path as needed
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    return df

data = load_data()

# Data preparation
def prepare_data(df):
    # Combine all years into one dataset
    columns_to_use = [
        "IKS 2021", "IKE 2021", "IKL 2021",
        "IKS 2022", "IKE 2022", "IKL 2022",
        "IKS 2023", "IKE 2023", "IKL 2023",
        "IKS 2024", "IKE 2024", "IKL 2024"
    ]
    X = df[columns_to_use]
    y = df["STATUS IDM 2024"]  # Using 2024 status as labels for training

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_categorical, label_encoder

X, y, label_encoder = prepare_data(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(input_dim=X.shape[1], output_dim=y.shape[1])

# Train the model
st.title("Indeks Desa Membangun - Neural Network Classification")
st.write("Melakukan klasifikasi status IDM menggunakan Neural Network.")

st.subheader("Training Model")
with st.spinner("Training sedang berlangsung..."):
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=0)

st.success("Training selesai!")

# Visualize training process
st.subheader("Training Metrics")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot loss
ax[0].plot(history.history['loss'], label='Training Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

# Plot accuracy
ax[1].plot(history.history['accuracy'], label='Training Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

st.pyplot(fig)

# Predict for 2025
def predict_2025(df, model, label_encoder):
    X_2025 = df[[
        "IKS 2021", "IKE 2021", "IKL 2021",
        "IKS 2022", "IKE 2022", "IKL 2022",
        "IKS 2023", "IKE 2023", "IKL 2023",
        "IKS 2024", "IKE 2024", "IKL 2024"
    ]]

    scaler = MinMaxScaler()
    X_2025_scaled = scaler.fit_transform(X_2025)

    predictions = model.predict(X_2025_scaled)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    return predicted_labels

predictions_2025 = predict_2025(data, model, label_encoder)
# Calculate accuracy on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
st.subheader("Akurasi Model")
st.write(f"Akurasi pada data uji: {test_accuracy:.2%}")

# Display predictions
st.subheader("Prediksi Status IDM Tahun 2025")
data["Prediksi STATUS IDM 2025"] = predictions_2025
comparison_columns = [
    "NAMA DESA", "IKS 2021", "IKE 2021", "IKL 2021",
    "IKS 2022", "IKE 2022", "IKL 2022",
    "IKS 2023", "IKE 2023", "IKL 2023",
    "IKS 2024", "IKE 2024", "IKL 2024",
    "STATUS IDM 2024", "Prediksi STATUS IDM 2025"
]
st.dataframe(data[comparison_columns])

st.success("Proses prediksi selesai!")

# Load Data
def load_data():
    file_path = 'DataTrainingKabSerang.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    return df

data = load_data()



# Chatbot Functionality
@st.cache_resource
def initialize_chatbot():
    return pipeline("text-generation", model="gpt2", max_length=200)

chatbot = initialize_chatbot()

def get_policy_recommendation(status_idm):
    policies = {
        "Mandiri": "Fokus pada diversifikasi ekonomi dan hilirisasi produk lokal.",
        "Maju": "Tingkatkan akses ke infrastruktur dasar seperti pasar dan perbankan.",
        "Berkembang": "Fokus pada pendidikan, kesehatan, dan peningkatan kualitas lingkungan.",
        "Tertinggal": "Prioritaskan pembangunan infrastruktur dasar dan tanggap bencana."
    }
    return policies.get(status_idm, "Belum ada rekomendasi spesifik.")

