import os
import shutil
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# =========================
# STEP 1: Load CSV
# =========================

print("Loading CSV...")

df = pd.read_csv("match_data.csv")

print(f"Total Matches: {len(df)}")

# =========================
# STEP 2: Convert CSV Rows to Text
# =========================

documents = []

for _, row in df.iterrows():

    text = (
        f"Match Date: {row['Date']}. "
        f"Team 1: {row['Team1']}. "
        f"Team 2: {row['Team2']}. "
        f"Venue: {row['Venue']}. "
        f"Result: {row['Result']}."
    )

    documents.append(text)

print("\nSample Document:")
print(documents[0])

# =========================
# STEP 3: Load Embedding Model
# =========================

print("\nLoading BGE-M3 model...")

try:
    model = SentenceTransformer(
        "BAAI/bge-m3",
        trust_remote_code=True
    )

    print("Model Loaded Successfully!")
    print(
        "Embedding Dimension:",
        model.get_sentence_embedding_dimension()
    )

except Exception as e:
    print("\nFailed to load model:")
    print(e)
    exit()

# =========================
# STEP 4: Generate Embeddings
# =========================

print("\nGenerating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings Shape:", embeddings.shape)

# =========================
# STEP 5: Recreate ChromaDB
# =========================

print("\nCreating ChromaDB...")

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="cricket_matches"
)

# =========================
# STEP 6: Store Data
# =========================

print("\nStoring Data...")

collection.add(
    ids=[str(i) for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings.tolist()
)

print("Data Stored Successfully!")

# =========================
# STEP 7: Search Function
# =========================

def search_matches(query, top_k=5):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results["documents"][0]

# =========================
# STEP 8: Interactive Search
# =========================

print("\n🏏 Cricket Match Search System")
print("Type 0 to Exit")

while True:

    query = input("\nAsk Question: ")

    if query.strip() == "0":
        print("Goodbye!")
        break

    try:

        results = search_matches(query)

        print("\nTop Results")
        print("-" * 50)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(result)

    except Exception as e:
        print("\nSearch Error:")
        print(e)