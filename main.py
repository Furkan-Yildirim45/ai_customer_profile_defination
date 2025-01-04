import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def classify_clusters_by_spending(customer_summary):
    # Kümeleme sonuçlarını alalım
    customer_summary['total_order_value'] = customer_summary['total_order_value'].astype(float)
    customer_summary['age'] = customer_summary['age'].astype(float)
    customer_summary['basket_size'] = customer_summary['basket_size'].astype(float)
    customer_summary['purchase_frequency'] = customer_summary['purchase_frequency'].astype(float)

    # Kümeleme sonuçlarını almak
    cluster_0 = customer_summary[customer_summary['Cluster'] == 0]
    cluster_1 = customer_summary[customer_summary['Cluster'] == 1]
    cluster_2 = customer_summary[customer_summary['Cluster'] == 2]

    # Küme sayıları
    print(f"\n0. Küme sayısı: {len(cluster_0)}")
    print(f"1. Küme sayısı: {len(cluster_1)}")
    print(f"2. Küme sayısı: {len(cluster_2)}")

    # Küme 0 (Yüksek harcama gibi) genel özellikleri
    print("\nKüme 0 Genel Özellikler:")
    print(f"Ortalama Yaş = {cluster_0['age'].mean():.2f}")
    print(f"Ortalama Toplam Harcama = {cluster_0['total_order_value'].mean():.2f}")
    print(f"Ortalama Sepet Boyutu = {cluster_0['basket_size'].mean():.2f}")
    print(f"Ortalama Satın Alma Sıklığı = {cluster_0['purchase_frequency'].mean():.2f}")

    # Küme 1 (Orta harcama gibi) genel özellikleri
    print("\nKüme 1 Genel Özellikler:")
    print(f"Ortalama Yaş = {cluster_1['age'].mean():.2f}")
    print(f"Ortalama Toplam Harcama = {cluster_1['total_order_value'].mean():.2f}")
    print(f"Ortalama Sepet Boyutu = {cluster_1['basket_size'].mean():.2f}")
    print(f"Ortalama Satın Alma Sıklığı = {cluster_1['purchase_frequency'].mean():.2f}")

    # Küme 2 (Düşük harcama gibi) genel özellikleri
    print("\nKüme 2 Genel Özellikler:")
    print(f"Ortalama Yaş = {cluster_2['age'].mean():.2f}")
    print(f"Ortalama Toplam Harcama = {cluster_2['total_order_value'].mean():.2f}")
    print(f"Ortalama Sepet Boyutu = {cluster_2['basket_size'].mean():.2f}")
    print(f"Ortalama Satın Alma Sıklığı = {cluster_2['purchase_frequency'].mean():.2f}")

if __name__ == "__main__":
    print("Hazırız modeli eğitmek için!")

    # Müşteri verisini yükleme
    customer_data = pd.read_json("data/customers.json")  # customer.json verisi

    # Satın alma verisini yükleme
    purchase_data = pd.read_json("data/purchase.json")  # purchase.json verisi

    # Satın alma verisini müşteri verisiyle birleştirme (customer_id ile)
    merged_data = pd.merge(purchase_data, customer_data, on="customer_id", how="inner")
    
    # Dolar işareti ve virgül gibi karakterleri temizleme
    merged_data["total_order_value"] = merged_data["total_order_value"].replace(
        {"\$": "", ",": ""}, regex=True)
    
    # İlk birkaç satırı kontrol etme
    print("Dönüştürülmeden önce:")
    print(merged_data["total_order_value"].head())  # Orijinal veri

    # Sayısal formata dönüştürme
    merged_data["total_order_value"] = pd.to_numeric(merged_data["total_order_value"], errors="coerce")

    # Sayısal verilerin doğru dönüştürülüp dönüştürülmediğini kontrol etme
    print("Dönüştürülmüş veriler:")
    print(merged_data["total_order_value"].head())  # Dönüştürülmüş veri

    # NaN değerleri kontrol etme
    if merged_data["total_order_value"].isnull().any():
        print("NaN değerler var. Düzeltme yapılmalı.")
    
    # Her müşteri için toplam harcama, sepet boyutu, ve satın alma sıklığını hesaplama
    customer_summary = merged_data.groupby("customer_id").agg(
        total_order_value=("total_order_value", "sum"),  # Toplam harcama
        basket_size=("basket_size", "mean"),  # Ortalama sepet boyutu
        purchase_frequency=("order_id", "count")  # Satın alma sıklığı
    ).reset_index()

    # Müşteri bilgilerini birleştirme (müşteri yaşı ve diğer bilgileri ekleme)
    customer_summary = pd.merge(customer_summary, customer_data[["customer_id", "age"]], on="customer_id", how="left")

    # K-Means için kullanılacak özellikler
    features = customer_summary[["age", "total_order_value", "basket_size", "purchase_frequency"]]  # Yaş, toplam harcama, sepet boyutu, satın alma sıklığı

    # Veriyi temizleme: Eksik değerleri kontrol etme
    features = features.dropna()  # NaN değerleri kaldırma

    # Veriyi standardize etme (K-Means daha iyi çalışır)
    scaler = StandardScaler()

    # Verinin sayısal olduğundan emin olun, verilerin türünü kontrol et
    # features.columns: ["age", "total_order_value", "basket_size", "purchase_frequency"]
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            print(f"{col} sütunu sayısal olmayan verilere sahip.")
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Veriyi standardize etme (K-Means için gereklidir)
    scaled_features = scaler.fit_transform(features)


    # K-Means ile kümeleme (3 küme ile)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Kümeleri veriye ekleme
    customer_summary["Cluster"] = clusters

    # Kümelerin görselleştirilmesi
    plt.scatter(customer_summary["age"], customer_summary["total_order_value"], c=customer_summary["Cluster"], cmap="viridis")
    plt.title(f"Customer Segments (K=3)")
    plt.xlabel("Age")
    plt.ylabel("Total Order Value")
    plt.show()

    # Kümeleri görüntüleme
    print(customer_summary.head()) 

    #Küme İçindeki Ortalama Değerleri Görüntüleme
    print(customer_summary.groupby("Cluster").mean())

    #4. Sonuçları Kaydetme ve Paylaşma
    customer_summary.to_csv("data/segments/customer_cluster_segment.csv", index=False)

    #modeli kaydet
    joblib.dump(kmeans, "models/kmeans_model.pkl")

    classify_clusters_by_spending(customer_summary)