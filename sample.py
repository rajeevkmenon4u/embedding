import os
import csv
import chromadb
from chromadb.utils import embedding_functions

# --- Step 1: Initialize Chroma persistent client ---
client = chromadb.PersistentClient(path="chromadb_store")

# --- Step 2: Define OpenAI embedding function ---
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"  # fast, high quality, low cost
)

# --- Step 3: Create or get Chroma collection ---
collection = client.get_or_create_collection(
    name="schema_collection",
    embedding_function=openai_ef
)

# --- Step 4: Read CSV and prepare text for embedding ---
csv_path = "schema.csv"
documents, ids, metadatas = [], [], []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        table = row["TableName"].strip()
        column = row["ColumnName"].strip()
        dtype = row["DataType"].strip()
        desc = row["Description"].strip()

        # Combine structured data into a semantic sentence
        text = f"Table {table}, column {column} ({dtype}): {desc}."
        documents.append(text)
        ids.append(f"row_{i}")
        metadatas.append({
            "table": table,
            "column": column,
            "datatype": dtype,
            "description": desc
        })

# --- Step 5: Add to ChromaDB (embedding happens automatically) ---
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ {len(documents)} rows embedded and stored in ChromaDB!")

# --- Step 6: Example query ---
query = "Which columns are used to identify users?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

# --- Step 7: Display results ---
print("\n🔍 Top matching schema columns:")
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"→ {meta['table']}.{meta['column']} ({meta['datatype']})")
    print(f"   Description: {meta['description']}")
    print(f"   Similarity Score: {1 - dist:.4f}\n")




✅ 4 rows embedded and stored in ChromaDB!

🔍 Top matching schema columns:
→ h1_user.user_id (int)
   Description: unique user identifier
   Similarity Score: 0.9142

→ h1_workflow_task.task_id (int)
   Description: unique identifier for workflow
   Similarity Score: 0.8713

# data reterival

from chromadb.utils import embedding_functions
import chromadb, os

client = chromadb.PersistentClient(path="chromadb_store")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = client.get_collection(
    name="schema_collection",
    embedding_function=openai_ef
)

query = "Where is the email address of the user stored?"
results = collection.query(query_texts=[query], n_results=2)

for meta in results["metadatas"][0]:
    print(f"{meta['table']}.{meta['column']} → {meta['description']}")



#  embedding 

     
  collection = client.get_or_create_collection(
    name="schema_collection",
    embedding_function=openai_ef
)


collection.add(
    documents=["Table h1_user, column user_id (int): unique user identifier."],
    ids=["row_1"],
    metadatas=[{
        "table": "h1_user",
        "column": "user_id",
        "datatype": "int",
        "description": "unique user identifier"
    }]
)


#filtering

query = "Which column stores user contact info?"

results = collection.query(
    query_texts=[query],
    n_results=3,
    where={"table": "h1_user"}  # 👈 filter by metadata
)

print("🔍 Filtered Results:")
for meta in results["metadatas"][0]:
    print(f"{meta['table']}.{meta['column']} — {meta['description']}")
            
                                 
