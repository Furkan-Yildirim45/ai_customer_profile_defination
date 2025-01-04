import os
import sys
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.utils.classify_clusters_by_spending import classify_clusters_by_spending

# Veriyi yükleme
def load_data(customer_file, purchase_file):
    customer_data = pd.read_json(customer_file)
    purchase_data = pd.read_json(purchase_file)
    return customer_data, purchase_data

# Veriyi birleştirme ve temizleme
def preprocess_data(customer_data, purchase_data):
    merged_data = pd.merge(purchase_data, customer_data, on="customer_id", how="inner")
    merged_data["total_order_value"] = merged_data["total_order_value"].replace({"\$": "", ",": ""}, regex=True)
    merged_data["total_order_value"] = pd.to_numeric(merged_data["total_order_value"], errors="coerce")

    if merged_data["total_order_value"].isnull().any():
        print("NaN değerler var. Düzeltme yapılmalı.")

    return merged_data

# Özet veri oluşturma
def create_customer_summary(merged_data, customer_data):
    customer_summary = merged_data.groupby("customer_id").agg(
        total_order_value=("total_order_value", "sum"),
        basket_size=("basket_size", "mean"),
        purchase_frequency=("order_id", "count")
    ).reset_index()

    customer_summary = pd.merge(customer_summary, customer_data[["customer_id", "age"]], on="customer_id", how="left")
    return customer_summary

# Veriyi standardize etme
def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Modeli eğitme
def train_kmeans(scaled_features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    return kmeans, clusters

# Kümeleri görselleştirme
def visualize_clusters(customer_summary):
    plt.scatter(
        customer_summary["age"], 
        customer_summary["total_order_value"], 
        c=customer_summary["Cluster"], 
        cmap="viridis"
    )
    plt.title("Customer Segments (K=3)")
    plt.xlabel("Age")
    plt.ylabel("Total Order Value")
    plt.show()

# Ana işlev
def train_model():
    print("Hazırız modeli eğitmek için!")

    # Verileri yükle
    customer_data, purchase_data = load_data("data/customers.json", "data/purchase.json")

    # Veriyi birleştir ve temizle
    merged_data = preprocess_data(customer_data, purchase_data)

    # Müşteri özetini oluştur
    customer_summary = create_customer_summary(merged_data, customer_data)

    # Özellikleri seç ve NaN değerleri temizle
    features = customer_summary[["age", "total_order_value", "basket_size", "purchase_frequency"]].dropna()

    # Sayısal verileri doğrula
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            print(f"{col} sütunu sayısal olmayan verilere sahip.")
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Özellikleri standardize et
    scaled_features = scale_features(features)

    # Modeli eğit ve kümeleri al
    kmeans, clusters = train_kmeans(scaled_features)

    # Kümeleri müşteri özetine ekle
    customer_summary["Cluster"] = clusters

    # Kümeleri görselleştir
    visualize_clusters(customer_summary)

    # Kümeleri analiz et ve kaydet
    print(customer_summary.groupby("Cluster").mean())
    customer_summary.to_csv("data/segments/customer_cluster_segment.csv", index=False)

    # Modeli kaydet
    joblib.dump(kmeans, "models/kmeans_model.pkl")

    # Kümeleme sonuçlarını sınıflandır
    classify_clusters_by_spending(customer_summary)

if __name__ == "__main__":
    train_model()
