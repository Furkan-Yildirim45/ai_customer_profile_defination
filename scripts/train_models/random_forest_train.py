from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Veri Yükleme Fonksiyonu
def load_data():
    customer_data = pd.read_json("data/customers.json")
    purchase_data = pd.read_json("data/purchase.json")
    customer_cluster_segment_data = pd.read_json("data/segments/customer_cluster_segment.json")
    return customer_data, purchase_data, customer_cluster_segment_data

# Veri Ön İşleme Fonksiyonu
def preprocess_data(purchase_data, customer_data):
    # Satın alma verilerini temizleme
    purchase_data["total_order_value"] = purchase_data["total_order_value"].replace({"\$": "", ",": ""}, regex=True)
    purchase_data["product_price"] = purchase_data["product_price"].replace({"\$": "", ",": ""}, regex=True)
    purchase_data["total_order_value"] = pd.to_numeric(purchase_data["total_order_value"], errors="coerce")
    purchase_data["product_price"] = pd.to_numeric(purchase_data["product_price"], errors="coerce")
    purchase_data["order_date"] = pd.to_datetime(purchase_data["order_date"], format="%m/%d/%Y")
    customer_data["registered_date"] = pd.to_datetime(customer_data["registered_date"], format="%m/%d/%Y")

# Hedef Kategoriyi Oluşturma
def create_target_category(purchase_data, target_category="Jewelry"):
    purchase_data["target_category_purchased"] = (purchase_data["product_category"] == target_category).astype(int)

# Müşteri Geçmişi ve Özellikler
def create_customer_features(purchase_data, customer_data, customer_cluster_segment_data):
    # Müşteri bazında hedef değişken
    target_purchase_summary = purchase_data.groupby("customer_id").agg(
        target_purchased=("target_category_purchased", "max")
    ).reset_index()

    # Müşteri geçmişi
    purchase_summary = purchase_data.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        avg_order_value=("total_order_value", "mean"),
        favorite_category=("product_category", lambda x: x.mode()[0]),
        last_order_date=("order_date", "max")
    ).reset_index()

    # Son 6 ay sipariş sayısı
    purchase_summary["orders_last_6_months"] = purchase_summary["last_order_date"].apply(
        lambda x: (pd.Timestamp.now() - x).days <= 6 * 30  # 6 ayı gün cinsinden hesapla (yaklaşık 180 gün)
    )
    purchase_summary["orders_last_6_months"] = purchase_summary["orders_last_6_months"].astype(int)

    # One-hot encoding işlemi
    purchase_summary = pd.get_dummies(purchase_summary, columns=["favorite_category"], drop_first=True)
    purchase_summary["days_since_last_order"] = (pd.Timestamp.now() - purchase_summary["last_order_date"]).dt.days

    # Son 6 Aydaki Harcamalar
    recent_purchase_data = purchase_data[purchase_data["order_date"] > pd.Timestamp.now() - pd.DateOffset(months=6)]
    recent_monetary_value = recent_purchase_data.groupby("customer_id")["total_order_value"].sum().reset_index()
    recent_monetary_value = recent_monetary_value.rename(columns={"total_order_value": "recent_monetary_value"})

    # Harcama varyansı hesaplama
    purchase_data["avg_order_value"] = purchase_data.groupby("customer_id")["total_order_value"].transform("mean")
    purchase_data["spending_variance"] = (purchase_data["total_order_value"] - purchase_data["avg_order_value"])**2
    spending_variance = purchase_data.groupby("customer_id")["spending_variance"].mean().reset_index()
    spending_variance = spending_variance.rename(columns={"spending_variance": "spending_variance"})

    # Kategori bazında harcama oranı
    purchase_data['category_spending'] = purchase_data.groupby('product_category')['total_order_value'].transform('sum')
    purchase_data['total_spending'] = purchase_data.groupby('customer_id')['total_order_value'].transform('sum')
    purchase_data['category_spending_ratio'] = purchase_data['category_spending'] / purchase_data['total_spending'] * 100

    category_spending_ratio = purchase_data.groupby("customer_id")["category_spending_ratio"].max().reset_index()
    category_spending_ratio = category_spending_ratio.rename(columns={"category_spending_ratio": "category_spending_ratio"})

    # Veri birleştirme
    merged_data = pd.merge(customer_data, purchase_summary, on="customer_id", how="left")
    merged_data = pd.merge(merged_data, target_purchase_summary, on="customer_id", how="left")
    merged_data = pd.merge(merged_data, category_spending_ratio, on="customer_id", how="left")
    merged_data["total_orders"] = merged_data["total_orders"].fillna(0)
    merged_data["avg_order_value"] = merged_data["avg_order_value"].fillna(0)
    merged_data["registered_days"] = (pd.Timestamp.now() - merged_data["registered_date"]).dt.days
    merged_data["cluster"] = customer_cluster_segment_data["cluster"]
    merged_data["target_purchased"] = merged_data["target_purchased"].fillna(0)
    merged_data["order_frequency"] = merged_data["total_orders"] / merged_data["registered_days"]
    merged_data["order_frequency"] = merged_data["order_frequency"].fillna(0)
    merged_data = pd.merge(merged_data, spending_variance, on="customer_id", how="left")

    return merged_data

# Özellik Seçimi
def select_features(merged_data):
    features = merged_data[[ 
        "age", 
        "total_orders", 
        "avg_order_value", 
        "registered_days", 
        "cluster",
        "order_frequency", 
        "spending_variance",  
        "category_spending_ratio",  
    ] + [col for col in merged_data.columns if col.startswith("favorite_category_")]]
    
    target = merged_data["target_purchased"]
    return features, target

# Modeli Eğitme ve Değerlendirme
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # SMOTE Uygulama
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Orijinal veri seti boyutları: {X_train.shape}, {y_train.value_counts().to_dict()}")
    print(f"SMOTE sonrası veri seti boyutları: {X_train_resampled.shape}, {np.bincount(y_train_resampled)}")

    # Model Eğitimi
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, min_samples_leaf=4, max_features="log2",
                                   min_samples_split=7, class_weight={0: 45, 1: 55})
    model.fit(X_train_resampled, y_train_resampled)

    # Model Performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)   
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Doğruluk (Accuracy): {accuracy:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return model

# Ana Fonksiyon
def main():
    print("Model eğitimi için hazırız!")
    
    # Veri Yükleme
    customer_data, purchase_data, customer_cluster_segment_data = load_data()
    
    # Veri Ön İşleme
    preprocess_data(purchase_data, customer_data)
    
    # Hedef Kategori Oluşturma
    create_target_category(purchase_data)
    
    # Özellikleri ve Hedefi Hazırlama
    merged_data = create_customer_features(purchase_data, customer_data, customer_cluster_segment_data)
    features, target = select_features(merged_data)
    
    # Eğitim ve Test Setlerine Ayırma
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Model Eğitimi ve Değerlendirme
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Modeli Kaydetme
    joblib.dump(model, "models/category_purchase_prediction_model.pkl")
    print("Model başarıyla kaydedildi.")

# Çalıştırma
if __name__ == "__main__":
    main()
