# query.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import config

# Ініціалізація клієнтів
# Модель для створення ембеддингів запитів
emb_model = SentenceTransformer(config.EMBEDDING_MODEL)
#openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Клієнт для підключення до Qdrant
qdrant = QdrantClient(url=config.QDRANT_URL)

# Клієнт для взаємодії з моделлю з Hugging Face
# Цей клієнт використовується для генерації відповідей
hf_client = InferenceClient(
    model=config.HF_LLM_MODEL,
    token=config.HUGGINGFACE_API_TOKEN,
    timeout=60
)

def answer_question(question: str) -> str:
    # 1. Створення ембеддингу запиту
    print(f"Створення ембеддингу для запиту: '{question}'...")

    # Модель "intfloat/multilingual-e5-base" вимагає префікс "query: " для запитів
    q_vector = emb_model.encode([f"query: {question}"], normalize_embeddings=True)[0].tolist()

    # 2. Пошук у Qdrant (тільки dense-пошук)
    print(f"Пошук {config.TOP_K} найрелевантніших чанків у Qdrant...")
    # Здійснюємо векторний пошук (dense)
    hits = qdrant.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=("dense", q_vector),
        limit=config.TOP_K,     # Кількість результатів, які потрібно повернути
        with_payload=True  # Переконуємось, що корисне навантаження (текст) буде повернуто
    )
    
    # Додаємо логування результатів пошуку
    print("\nЗнайдені чанки:")
    for i, hit in enumerate(hits):
        print(f"--- Чанк #{i+1} (score: {hit.score:.4f}) ---")
        if hit.payload and "text" in hit.payload:
            print(hit.payload["text"][:200] + "...") # Виводимо перші 200 символів
    print("-" * 20)

    # 3. Формування контексту з результатів
    context = ""
    for i, chunk in enumerate(hits):
        if chunk.payload and "text" in chunk.payload:
            context += f"Джерело {i+1}:\n"
            context += chunk.payload["text"] + "\n\n"

    # Якщо не знайдено жодного чанка, повертаємо повідомлення
    if not context:
        return "На жаль, не вдалося знайти релевантну інформацію в базі знань."

    #context = "\n\n".join(hit.payload["text"] for hit in hits if hit.payload is not None and "text" in hit.payload)

    # 4. Створення промпту для моделі
    # `system_prompt` встановлює роль моделі та інструкції щодо генерації
    system_prompt = (
        "Ти — висококваліфікований технічний помічник. "
        "Відповідай стисло і тільки на основі НАДАНОГО КОНТЕКСТУ. "
        "Якщо у контексті немає відповіді — чесно скажи, що на основі наданої інформації відповісти неможливо. "
        "Не вигадуй інформацію."
    )
    
    # `user_prompt` містить наданий контекст і запитання
    user_prompt = (
        f"КОНТЕКСТ:\n---\n{context}\n---\n\n"
        f"ПИТАННЯ: {question}\n\n"
        "ВІДПОВІДЬ:"
    )

    # 5. Генерація відповіді через ChatCompletion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Звернення до Hugging Face для генерації відповіді
        resp = hf_client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1, # Низька температура робить відповідь більш детермінованою
            top_p=0.95
        )
        answer = resp.choices[0].message["content"].strip()
        return answer
    except Exception as e:
        print(f"Помилка при генерації відповіді: {e}")
        return "Виникла помилка під час звернення до мовної моделі."


    # chat_response = openai_client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "Ти - помічник, що відповідає на запитання користувача на основі наданого контексту."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.0,
    #     max_tokens=512
    # )

    # return chat_response.choices[0].message.content or ""

if __name__ == "__main__":
    import sys
    # Запуск скрипта з командного рядка
    # Перевіряємо, чи користувач взагалі ввів питання
    if len(sys.argv) > 1:
        # Правильно об'єднуємо всі слова запитання через пробіл
        question_input = " ".join(sys.argv[1:])
        
        print("\n--- ВІДПОВІДЬ МОДЕЛІ ---")
        print(answer_question(question_input))
    else:
        # Якщо питання не введено, виводимо інструкцію
        print("Будь ласка, вкажіть ваше питання як аргумент командного рядка.")
        print('Приклад: python query.py "Які ключові особливості задач пакування?"')