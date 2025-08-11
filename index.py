import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
#from openai import OpenAI
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf  # Бібліотека для розбору PDF-файлів
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

# Ініціалізація моделі для створення ембеддингів та клієнта Qdrant
# Ця модель (Sentence-Transformer) перетворює текст на числові вектори
emb_model = SentenceTransformer(config.EMBEDDING_MODEL)
# Підключення до сервера Qdrant
qdrant = QdrantClient(url=config.QDRANT_URL)  

def load_pdf(path: str) -> str:
    # partition_pdf розбиває PDF на логічні елементи (заголовки, текст, таблиці тощо)
    elements = partition_pdf(path)
    # З'єднуємо текст усіх елементів, які мають текстовий вміст
    return "\n".join([el.text for el in elements if hasattr(el, "text")])

def load_html(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    # get_text видаляє HTML-теги та повертає чистий текст
    return soup.get_text(separator="\n")

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,       # Максимальний розмір чанка
        chunk_overlap=config.CHUNK_OVERLAP  # Кількість символів, які перекриваються між сусідніми чанками
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    # Модель "intfloat/multilingual-e5-base" вимагає префікс "passage: " для документів
    passages = [f"passage: {c}" for c in chunks]
    # encode перетворює тексти на вектори. normalize_embeddings=True для косинусної подібності.
    vectors = emb_model.encode(passages, show_progress_bar=True, normalize_embeddings=True)
    return [v.tolist() for v in vectors]

def create_collection(vector_size: int):
    print(f"Пересоздаємо колекцію '{config.COLLECTION_NAME}'...")
    # recreates_collection видаляє існуючу колекцію та створює нов
    qdrant.recreate_collection(
        collection_name=config.COLLECTION_NAME,
        # Налаштування для щільних (dense) векторів - це наші ембеддинги
        vectors_config={
            "dense": rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE # Використовуємо косинусну відстань для порівняння векторів
            ),
        },
        # Налаштування для розріджених (sparse) векторів - для повнотекстового пошуку
        # У поточному коді query.py цей функціонал не використовується, але колекція підготовлена
        sparse_vectors_config={
            "sparse": rest.SparseVectorParams(index=rest.SparseIndexParams(on_disk=False))
        }
    )
    # Створення індексу для текстового поля. Це дозволяє здійснювати повнотекстовий пошук
    # Цей індекс також поки не використовується в query.py, гібридний пошук поки не реалізований
    qdrant.create_payload_index(
        collection_name=config.COLLECTION_NAME,
        field_name="text",
        field_schema=rest.TextIndexParams(
            type=rest.TextIndexType.TEXT,
            tokenizer=rest.TokenizerType.WHITESPACE,
            min_token_len=2,
            max_token_len=15,
            lowercase=True,
        )
    )
    print("Колекція створена.")

def index_document(path: str, is_pdf: bool = True):
    text = load_pdf(path) if is_pdf else load_html(path)
    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)
    create_collection(len(vectors[0]))
    
    # Формування точок для завантаження в Qdrant
    points = [
        rest.PointStruct(id=i, 
                         vector={"dense": vectors[i]},  # Додаємо ембедінг
                         payload={"text": chunks[i]})   # Додаємо оригінальний текст чанка як корисне навантаження
        for i in range(len(chunks))
    ]
    # Завантаження точок у колекцію Qdrant
    qdrant.upsert(
        collection_name=config.COLLECTION_NAME,
        points=points
    )
    print(f"Indexed {len(chunks)} chunks from {path}.")

if __name__ == "__main__":
    # Запуск скрипта з командного рядка
    import sys
    # sys.argv[1] - шлях до файлу, sys.argv[2] - тип файлу (наприклад, "pdf")
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
        fmt = sys.argv[2].lower() == "pdf" if len(sys.argv) > 2 else True
        index_document(doc_path, is_pdf=fmt)
    else:
        print("Будь ласка, вкажіть шлях до файлу.")
        print('Приклад: python index.py "path/to/document.pdf" pdf')