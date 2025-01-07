import os
import joblib
import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from scripts.train_models.k_means_train_model import (
    create_customer_summary,
    preprocess_data,
    scale_features,
    train_kmeans,
    visualize_clusters,
)
from scripts.train_models.random_forest_train import (
    create_customer_features,
    create_target_category,
    preprocess_data_random_forest,
    random_forest_model_train,
    select_features,
    train_and_evaluate_model,
)
from scripts.utils.classify_clusters_by_spending import classify_clusters_by_spending


class CustomerSegmentationCLI:
    def __init__(self):
        self.customers_data = pd.read_json("data/customers.json")
        self.purchase_data = pd.read_json("data/purchase.json")
        self.customer_cluster_segments_data = pd.read_json(
            "data/segments/customer_cluster_segment.json"
        )
        self.k_means_model = joblib.load("models/kmeans_model.pkl")
        self.random_forest_model = joblib.load(
            "models/category_purchase_prediction_model.pkl"
        )

    def main_menu(self):
        while True:
            print("\n--- Ana Menü ---")
            print("Info: Hazır olarak modelimiz ve verimiz bulunmaktadır.")
            print("1. Model Yükle")
            print("2. Veri Yükle (JSON)")
            print("3. Model Eğit")
            print("4. Müşteri Gruplandır (Segmentasyon)")
            print("5. Satın Alma Tahmini Yap")
            print("6. Kategori Önerisi Yap")
            print("7. Kategori Analizi Yap")
            print("8. Çıkış")
            choice = input("Seçiminizi yapın: ")

            if choice == "1":
                self.load_models()
            elif choice == "2":
                self.load_data()
            elif choice == "3":
                self.train_model()
            elif choice == "4":
                self.segment_customers()
            elif choice == "5":
                self.predict_purchase()
            elif choice == "6":
                self.recommend_category()
            elif choice == "7":
                self.analyze_category()
            elif choice == "8":
                print("Çıkılıyor...")
                break
            else:
                print("Geçersiz seçim!")

    def load_models(self):
        print("\nModel Yükleme")
        print(
            "Müşteri segmentasyonu için **K-Means** ve satın alma tahmini için **Random Forest** modellerine ihtiyacınız var."
        )
        print("Lütfen iki modelin dosya yollarını belirtin.\n")

        # Yüklenen modelleri kayıt etmek için bir klasör oluştur
        saved_models_dir = "input_loaded_models"
        os.makedirs(saved_models_dir, exist_ok=True)

        # K-Means modelini yükle
        kmeans_path = input(
            "K-Means model dosyasının yolunu girin (örnek: k_means_model.pkl): "
        )
        try:
            with open(kmeans_path, "rb") as file:
                self.k_means_model = joblib.load(kmeans_path)
                print("K-Means modeli başarıyla yüklendi.")

                # Modeli kayıt et
                kmeans_save_path = os.path.join(saved_models_dir, "k_means_model.pkl")
                joblib.dump(self.k_means_model, kmeans_save_path)
                print(f"K-Means modeli {kmeans_save_path} konumuna kaydedildi.")
        except FileNotFoundError:
            print("K-Means model dosyası bulunamadı. Lütfen geçerli bir yol girin.")
            self.k_means_model = None

        # Random Forest modelini yükle
        rf_path = input(
            "Random Forest model dosyasının yolunu girin (örnek: random_forest_model.pkl): "
        )
        try:
            with open(rf_path, "rb") as file:
                self.random_forest_model = joblib.load(file)
                print("Random Forest modeli başarıyla yüklendi.")

                # Modeli kayıt et
                rf_save_path = os.path.join(saved_models_dir, "random_forest_model.pkl")
                joblib.dump(self.random_forest_model, rf_save_path)
                print(f"Random Forest modeli {rf_save_path} konumuna kaydedildi.")
        except FileNotFoundError:
            print(
                "Random Forest model dosyası bulunamadı. Lütfen geçerli bir yol girin."
            )
            self.random_forest_model = None

    def load_data(self):
        print("\n--- Veri Yükleme ---")
        print(
            "Müşteri segmentasyonu için **customers**, satın alma tahmini için **purchase** ve müşteri gruplarını saklamak için **customer_cluster_segments_data** adlı verilere ihtiyacınız var."
        )
        print("Veriler JSON formatında olmalıdır.")
        print(
            "Örnek ve hazır veriler klasöründe bulunmaktadır. (örnek: data/customers.json, data/purchase.json)"
        )

        # Yüklenen modelleri kayıt etmek için bir klasör oluştur
        loaded_datas_dir = "data/input_loaded_datas"
        os.makedirs(loaded_datas_dir, exist_ok=True)

        # Customers verisi yükle
        print("\n** Customers Verisi **")
        print(
            "Customers verisi şu sütunları içermelidir: ['customer_id', 'first_name', 'last_name', 'gender', 'age', 'registered_date', 'city','country']"
        )
        customers_path = input(
            "Customers veri dosyasının yolunu girin (örnek: customers.json): "
        )
        try:
            with open(
                customers_path, "r", encoding="utf-8"
            ) as file:  # UTF-8 kodlamasıyla dosyayı açıyoruz
                customers_data = json.load(file)
                self.customers_data = pd.DataFrame(customers_data)
                print("Customers verisi başarıyla yüklendi.")
                print(self.customers_data.head())

                # Veriyi kaydet
                self.customers_data.to_json(
                    "data/input_loaded_datas/customers.json",
                    orient="records",
                    lines=False,
                )
                print(
                    "Customers verisi 'data/input_loaded_datas/customers.json' dosyasına kaydedildi."
                )
        except FileNotFoundError:
            print("Customers veri dosyası bulunamadı. Lütfen geçerli bir yol girin.")
            self.customers_data = None

        # Purchase verisi yükle
        print("\n** Purchase Verisi **")
        print(
            "Purchase verisi şu sütunları içermelidir: ['customer_id', 'order_id', 'order_date', 'order_time','product_id', 'product_price', 'quantity', 'total_order_value','product_category', 'payment_method', 'basket_size']"
        )
        purchase_path = input(
            "Purchase veri dosyasının yolunu girin (örnek: data/purchase.json): "
        )
        try:
            with open(
                purchase_path, "r", encoding="utf-8"
            ) as file:  # UTF-8 kodlamasıyla dosyayı açıyoruz
                purchase_data = json.load(file)
                self.purchase_data = pd.DataFrame(purchase_data)
                print("Purchase verisi başarıyla yüklendi.")
                print(self.purchase_data.head())

                # Veriyi kaydet
                self.purchase_data.to_json(
                    "data/input_loaded_datas/purchase.json",
                    orient="records",
                    lines=False,
                )
                print(
                    "Customers verisi 'data/input_loaded_datas/purchase.json' dosyasına kaydedildi."
                )
        except FileNotFoundError:
            print("Purchase veri dosyası bulunamadı. Lütfen geçerli bir yol girin.")
            self.purchase_data = None

        # Customer Cluster Segments Data kontrolü
        print("\n** Customer Cluster Segments Verisi **")
        segments_path = input(
            "Customer Cluster Segments veri dosyasının yolunu girin (örnek: data/segments/customer_cluster_segments.json): "
        )

        if os.path.exists(segments_path):
            try:
                with open(
                    segments_path, "r", encoding="utf-8"
                ) as file:  # UTF-8 kodlamasıyla dosyayı açıyoruz
                    cluster_segments_data = json.load(file)
                    self.customer_cluster_segments_data = pd.DataFrame(
                        cluster_segments_data
                    )
                    print("Customer Cluster Segments verisi başarıyla yüklendi.")
                    print(self.customer_cluster_segments_data.head())

                    self.customer_cluster_segments_data.to_json(
                        "data/input_loaded_datas/customer_cluster_segments.json",
                        orient="records",
                        lines=False,
                    )
                    print(
                        "Customers verisi 'data/input_loaded_datas/customer_cluster_segments.json' dosyasına kaydedildi."
                    )
            except FileNotFoundError:
                print("Customer Cluster Segments veri dosyası okunamadı.")
                self.customer_cluster_segments_data = None
        else:
            print("Customer Cluster Segments verisi bulunamadı.")
            create_segments = input("Bu veri oluşturulsun mu? (E/H): ").lower()
            if create_segments == "e":
                self.create_customer_cluster_segments_data()

    def create_customer_cluster_segments_data(self):
        print("\nMüşteri Cluster Segments Verisi oluşturuluyor...")
        if self.customers_data is not None and self.purchase_data is not None:
            # Veriyi birleştir ve temizle
            merged_data = preprocess_data(self.customers_data, self.purchase_data)

            # Müşteri özetini oluştur
            customer_summary = create_customer_summary(merged_data, self.customers_data)

            # Özellikleri seç ve NaN değerlerini temizle
            features = customer_summary[
                ["age", "total_order_value", "basket_size", "purchase_frequency"]
            ].dropna()

            # Sayısal verileri doğrula
            for col in features.columns:
                if not np.issubdtype(features[col].dtype, np.number):
                    print(f"{col} sütunu sayısal olmayan verilere sahip.")
                else:
                    features[col] = pd.to_numeric(features[col], errors="coerce")

            # Özellikleri standardize et
            scaled_features = scale_features(features)

            # Kümeleri oluştur ve müşteri özetine ekle
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            customer_summary["Cluster"] = clusters

            # İstenilen veri formatını oluştur
            self.customer_cluster_segments_data = customer_summary[
                [
                    "customer_id",
                    "total_order_value",
                    "basket_size",
                    "purchase_frequency",
                    "age",
                    "Cluster",
                ]
            ]

            # Veriyi kaydet
            save_path = "data/input_loaded_datas/customer_cluster_segments_data.json"
            self.customer_cluster_segments_data.to_json(
                save_path, orient="records", lines=False
            )
            print(
                f"Müşteri Cluster Segments verisi başarıyla oluşturuldu ve {save_path} dosyasına kaydedildi."
            )
            print(self.customer_cluster_segments_data.head())
        else:
            print("Customers verisi yüklenmeden segment verisi oluşturulamaz.")
    def train_model(self):
        # Verilerin dolu olup olmadığını kontrol et
        if self.customers_data is None or self.purchase_data is None:
            print("Veriler eksik. Eğitim yapılamaz. Lütfen verileri kontrol edin.")
            return

        print("Modeller eğitiliyor...")

        # KMeans modelini eğitmek için verileri ölçeklendir ve temizle
        print("k_means_model eğitiliyor...")
        scaled_features, customer_summary = self.get_scaled_features()

        # KMeans modelini eğit
        kmeans, clusters = train_kmeans(scaled_features)
        customer_summary["Cluster"] = clusters

        # KMeans kümelerini görselleştir
        visualize_clusters(customer_summary)

        # KMeans modelini kaydet
        joblib.dump(kmeans, "models/custom_trained_models/kmeans_model.pkl")
        self.k_means_model = kmeans
        print("KMeans modeli başarıyla eğitildi ve kaydedildi.")

        # Random Forest modelini eğit
        print("random_forest_model eğitiliyor...")
        random_forest_model_train(
            self.purchase_data, self.customers_data, customer_summary
        )
        print("Random Forest modeli başarıyla eğitildi.")

    def get_scaled_features(self):
        if self.customers_data is None or self.purchase_data is None:
            raise ValueError("Veriler eksik. Lütfen müşteri ve satın alma verilerini yükleyin.")

        # Verileri birleştir ve temizle
        merged_data = preprocess_data(self.customers_data, self.purchase_data)
        customer_summary = create_customer_summary(merged_data, self.customers_data)

        # İlgili özellikleri seç ve ölçeklendir
        features = customer_summary[
            ["age", "total_order_value", "basket_size", "purchase_frequency"]
        ].dropna()
        scaled_features = scale_features(features)

        return scaled_features, customer_summary

    def segment_customers(self):
        if self.k_means_model is None:
            print("KMeans modeli bulunamadı. Lütfen önce modeli eğitin.")
            return

        if self.customers_data is None or self.purchase_data is None:
            print("Veriler eksik. Müşteri segmentasyonu yapılamaz.")
            return

        # Müşteri segmentasyonu için gerekli verileri birleştir ve temizle
        scaled_features, customer_summary = self.get_scaled_features()

        # Kümeleri tahmin et ve müşteri özetine ekle
        clusters = self.k_means_model.predict(scaled_features)
        customer_summary["Cluster"] = clusters

        # Segmentasyonu kaydet
        self.customer_cluster_segments_data = customer_summary[
            [
                "customer_id",
                "total_order_value",
                "basket_size",
                "purchase_frequency",
                "age",
                "Cluster",
            ]
        ]
        print("Müşteri segmentasyonu başarıyla oluşturuldu.")

        # Kullanıcıya kümeleri görme seçeneği sun
        show_clusters = input("Kümeleri görmek ister misiniz? (E/H): ").lower()
        if show_clusters == "e":
            classify_clusters_by_spending(customer_summary)



    def predict_purchase(self):
        if self.model is None:
            print("Lütfen önce modeli yükleyin veya eğitin!")
            return
        if self.data is None:
            print("Lütfen önce veri yükleyin!")
            return

        print("Satın alma tahminleri yapılıyor...")
        X = self.data[["recency", "frequency", "monetary"]]
        self.data["purchase_probability"] = self.model.predict_proba(X)[:, 1]
        print(self.data[["recency", "frequency", "monetary", "purchase_probability"]])

    def recommend_category(self):
        if self.data is None:
            print("Lütfen önce veri yükleyin!")
            return

        print("Kategori önerisi yapılıyor...")
        # Örnek: Recency ve Frequency değerlerine göre öneri (basit bir mantık)
        self.data["recommended_category"] = self.data["segment"].apply(
            lambda x: f"Kategori {x + 1}"
        )
        print("Öneriler tamamlandı:")
        print(self.data[["segment", "recommended_category"]])

    def analyze_category(self):
        if self.data is None:
            print("Lütfen önce veri yükleyin!")
            return

        print("Kategori analizi yapılıyor...")
        # Örnek analiz: Her segmentteki müşterilerin sayısı ve satın alma olasılıklarının ortalaması
        if "purchase_probability" not in self.data.columns:
            print("Satın alma tahmini yapılmadı. Lütfen önce satın alma tahmini yapın!")
            return

        analysis = self.data.groupby("segment").agg(
            customer_count=("segment", "count"),
            avg_purchase_probability=("purchase_probability", "mean"),
        )
        print("Analiz tamamlandı:")
        print(analysis)


if __name__ == "__main__":
    # Uygulamayı başlat
    app = CustomerSegmentationCLI()
    app.main_menu()
