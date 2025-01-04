import json

def read_json(file_path):
    """JSON dosyasını okur ve Python listesi olarak döner."""
    with open(file_path, 'r') as file:
        return json.load(file)

def update_order_ids(data, increment_value):
    """JSON verisindeki her bir order_id değerini artırır."""
    for entry in data:
        entry['order_id'] += increment_value
    return data

def write_json(file_path, data):
    """Düzenlenmiş JSON verisini yeni bir dosyaya kaydeder."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def calculate_total_order_value(data):
    """
    JSON verisi içindeki product_price, quantity, basket_size ve total_order_value sütunlarını
    düzenler ve tutarlı bir şekilde hesaplar.

    Args:
        data (list): JSON veri listesi (her biri bir sözlük).

    Returns:
        list: Güncellenmiş JSON veri listesi.
    """
    for entry in data:
        # product_price değerini $ işaretinden arındırıp float'a çeviriyoruz
        product_price = float(entry['product_price'].replace('$', ''))
        
        # quantity değeri sıfırsa 1 yapıyoruz
        quantity = entry['quantity']
        if quantity == 0:
            quantity = 1
        entry['quantity'] = quantity  # quantity'yi güncelle

        # basket_size en az quantity kadar olmalı
        basket_size = entry['basket_size']
        basket_size = max(basket_size, quantity)
        entry['basket_size'] = basket_size  # basket_size'yi güncelle

        # total_order_value başlangıç olarak quantity * product_price
        total_order_value = product_price * quantity

        # Eğer basket_size 1'den büyükse, minimum değer basket_size * product_price olabilir
        total_order_value = max(total_order_value, basket_size * product_price)

        # total_order_value'yu güncelle
        entry['total_order_value'] = f"${round(total_order_value, 2)}"  # Virgülden sonra 2 basamak

    return data

def combine_json_files(file_paths, output_file):
    """
    Birden fazla JSON dosyasını birleştirip tek bir JSON dosyası oluşturur.

    Args:
        file_paths (list): Birleştirilecek JSON dosyalarının yol listesi.
        output_file (str): Birleştirilen verilerin kaydedileceği dosya adı.

    Returns:
        None
    """
    combined_data = []

    try:
        # Her bir JSON dosyasını oku ve verileri birleştir
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                data = json.load(file)
                combined_data.extend(data)  # Listeyi birleştir

        # Birleştirilen verileri yeni bir dosyaya kaydet
        with open(output_file, 'w') as output:
            json.dump(combined_data, output, indent=4)
        
        print(f"JSON dosyaları başarıyla birleştirildi! Çıkış dosyası: {output_file}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

def main():
    file_name = "purchase6.json";
    input_file = 'data/'+file_name;      # Giriş JSON dosyası
    output_file = 'data/corrected_data/{0}'.format(file_name)  # Çıkış JSON dosyası
    increment_value = 5000            # order_id artış değeri

    try:
        print("JSON dosyası okunuyor...")
        data = read_json(input_file)

        print("order_id değerleri güncelleniyor...")
        updated_data = calculate_total_order_value(data)

        print("Düzenlenmiş veriler kaydediliyor...")
        write_json(output_file, updated_data)

        print(f"İşlem tamamlandı! Yeni dosya: {output_file}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

def combineMain():
    # Birleştirilecek JSON dosyalarının yollarını tanımlayın
    file_paths = [
        "data/corrected_data/purchase.json",
        "data/corrected_data/purchase2.json",
        "data/corrected_data/purchase3.json",
        "data/corrected_data/purchase4.json",
        "data/corrected_data/purchase5.json",
        "data/corrected_data/purchase6.json"
    ]
    # Çıkış dosyasını tanımlayın
    output_file = "data/purchase.json"

    # Birleştirme işlemini gerçekleştir
    combine_json_files(file_paths, output_file)


if __name__ == "__main__":
    # main()
    combineMain()
    
