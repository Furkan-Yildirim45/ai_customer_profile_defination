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
