import os
import csv
import chromadb
from chromadb.utils import embedding_functions

# --- Step 1: Initialize Chroma client (persistent) ---
client = chromadb.PersistentClient(path="chromadb_store")

# --- Step 2: Define embedding function using OpenAI ---
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# --- Step 3: Create or get Chroma collection ---
collection = client.get_or_create_collection(
    name="table_schema_embeddings",
    embedding_function=openai_ef
)

# --- Step 4: Read CSV file and combine columns into descriptive text ---
csv_path = "schema.csv"
documents, ids, metadatas = [], [], []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        table = row["TableName"].strip()
        column = row["ColumnName"].strip()
        dtype = row["DataType"].strip()
        desc = row.get("Description", "").strip()
        use = row.get("UseCase", "").strip()

        # Combine into one meaningful sentence
        text = (
            f"Table {table}, column {column} ({dtype}): {desc}. "
            f"Use case: {use}."
        )

        documents.append(text)
        ids.append(f"row_{i}")
        metadatas.append({
            "table": table,
            "column": column,
            "datatype": dtype,
            "description": desc,
            "usecase": use
        })

# --- Step 5: Store in ChromaDB ---
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ {len(documents)} schema entries embedded and stored!")

# --- Step 6: Query example ---
query = "Which columns are unique identifiers?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\n🔍 Query Results:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {meta['table']}.{meta['column']} ({meta['datatype']}): {meta['description']}")
    print(f"  🔹 {doc}\n")
