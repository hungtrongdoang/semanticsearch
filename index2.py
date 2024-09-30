import pyodbc
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pyvi import ViTokenizer

# Cấu hình chuỗi kết nối tới SQL Server
conn_str = (
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=DUNGHT\\DUNGHT;'  # Thay bằng tên máy chủ của bạn
    'DATABASE=dbTKB;'  # Thay bằng tên database của bạn
    'Trusted_Connection=yes;'  # Sử dụng Windows Authentication
    'TrustServerCertificate=yes;'  # Bỏ qua kiểm tra SSL
)

# Kết nối tới SQL Server
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Truy vấn dữ liệu từ bảng tbCanBo
query_canbo = """
    SELECT TOP (1000) [IdCanBo], [IdDonVi], [HoDem], [Ten], [TgTao], [TgSua], 
                      [NguoiTao], [NguoiSua], [TT], [GhiChu], [TonTai]
    FROM [dbTKB].[dbo].[tbCanBo]
"""
cursor.execute(query_canbo)
canbo_data = cursor.fetchall()
columns_canbo = [column[0] for column in cursor.description]  # Lấy tên cột từ bảng tbCanBo

# Truy vấn dữ liệu từ bảng tbHocVien
query_hocvien = """
    SELECT TOP (1000) [IdHocVien], [HoDem], [Ten], [IdLopHanhChinh], [MaHocVien]
    FROM [dbTKB].[dbo].[tbHocVien]
"""
cursor.execute(query_hocvien)
hocvien_data = cursor.fetchall()
columns_hocvien = [column[0] for column in cursor.description]  # Lấy tên cột từ bảng tbHocVien

# Đóng kết nối SQL sau khi truy vấn
cursor.close()
conn.close()

# Chuyển đổi dữ liệu thành DataFrame
df_canbo = pd.DataFrame.from_records(canbo_data, columns=columns_canbo)
df_hocvien = pd.DataFrame.from_records(hocvien_data, columns=columns_hocvien)

# Tạo kết nối Elasticsearch
client = Elasticsearch(hosts=["http://localhost:9200/"])
index_name = "demo_simcse_v2"

# Tải model embedding
model_embedding = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

# Hàm để mã hóa văn bản
def embed_text(batch_text):
    batch_embedding = model_embedding.encode(batch_text)
    return [vector.tolist() for vector in batch_embedding]

# Hàm để xử lý văn bản từ nhiều cột thành một chuỗi
def combine_text_for_indexing(row, table_name):
    if table_name == 'canbo':
        return f"{row['HoDem']} {row['Ten']} {row['IdCanBo']} {row['IdDonVi']} {row['GhiChu'] or ''}"
    elif table_name == 'hocvien':
        return f"{row['HoDem']} {row['Ten']} {row['MaHocVien']}"

# Hàm để đẩy batch dữ liệu lên Elasticsearch
def index_batch(docs):
    requests = []
    titles = [ViTokenizer.tokenize(doc["title"]) for doc in docs]
    title_vectors = embed_text(titles)
    
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = index_name
        request["title_vector"] = title_vectors[i]  # Vector từ SimCSE
        requests.append(request)
    
    bulk(client, requests)

# Xóa index nếu đã tồn tại và tạo mới
client.indices.delete(index=index_name, ignore=[404])
client.indices.create(index=index_name, body={
    "mappings": {
        "properties": {
            "title_vector": {"type": "dense_vector", "dims": 768},
            "id": {"type": "keyword"},
            "title": {"type": "text"}
        }
    }
})

# Chuyển đổi dữ liệu từ DataFrame thành danh sách các documents để index
batch_size = 128
docs = []
count = 0

# Xử lý dữ liệu từ bảng tbCanBo
for index, row in df_canbo.iterrows():
    count += 1
    item = {
        'id': row['IdCanBo'],
        'title': combine_text_for_indexing(row, 'canbo')
    }
    docs.append(item)
    
    if count % batch_size == 0:
        index_batch(docs)
        docs = []
        print("Indexed {} documents from CanBo.".format(count))

# Xử lý dữ liệu từ bảng tbHocVien
for index, row in df_hocvien.iterrows():
    count += 1
    item = {
        'id': row['IdHocVien'],
        'title': combine_text_for_indexing(row, 'hocvien')
    }
    docs.append(item)
    
    if count % batch_size == 0:
        index_batch(docs)
        docs = []
        print("Indexed {} documents from HocVien.".format(count))

# Index phần dữ liệu còn lại nếu chưa đủ batch_size
if docs:
    index_batch(docs)
    print("Indexed {} documents.".format(count))

client.indices.refresh(index=index_name)
print("Done indexing.")
