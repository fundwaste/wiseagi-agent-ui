import textwrap
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from datetime import datetime

# Load embedding model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embedding
def generate_embedding(text):
    return model.encode([text])[0].tolist()

# Query OpenAI
def query_openai_context(prompt, context):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=500,
        messages=[
            {"role": "system", "content": "You are a helpful professor who always provides sources."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

# Perform the full vector + GPT query
def query_collection_openai(user_query, collection: Collection, top_k=5):
    query_vector = generate_embedding(user_query)

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 5}
    }
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["Text"]
    )

    context_texts = [
        r.entity.get("Text") for r in results[0] if r.entity.get("Text")
    ]
    context = " ".join(context_texts)

    if not context:
        return "No useful context was found in the database.", ""

    answer = query_openai_context(user_query, context)
    return answer, context
