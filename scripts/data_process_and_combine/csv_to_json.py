import json
import pandas as pd

# 1. CSV dosyasını yükleyin
csv_file_path = "data/segments/customer_cluster_segment.csv"  # CSV dosyanızın yolu
df = pd.read_csv(csv_file_path)

# 2. Veriyi bir liste şeklinde hazırlayın
records_list = df.to_dict(orient="records")

# 3. Listeyi JSON formatında dosyaya yazın
json_file_path = "data/segments/customers.json"  # JSON dosyasının kaydedileceği yer
with open(json_file_path, "w") as json_file:
    json.dump(records_list, json_file, indent=4)

print("CSV'den JSON'a dönüşüm işlemi tamamlandı!")
