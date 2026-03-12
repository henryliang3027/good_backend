import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "./drink_vector_db"
COLLECTION_NAME = "drink_catalog"

model = SentenceTransformer("clip-ViT-B-32")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def search_by_text(query: str, n_results: int = 3):
    text_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[text_emb],
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    return results


def main():
    queries = ["橘色飲料"]

    total = collection.count()
    print(f"資料庫共 {total} 筆商品\n")

    for query in queries:
        print(f"查詢：「{query}」")
        results = search_by_text(query, n_results=min(1, total))
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            name = f"{meta.get('color', '')}{meta.get('display_name', '')}"
            print(f"  {name}  distance={dist:.4f}")
        print()


if __name__ == "__main__":
    main()
