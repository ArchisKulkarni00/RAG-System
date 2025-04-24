from utils import load_config, read_text_files, semantic_chunker, generate_embeddings, store_in_milvus
from ollama import Client
import hashlib
import numpy as np

config = load_config()
print("⚙️ Loaded configs:", config)

# SETUP OLLAMA CLIENT
ollama_url = 'http://{host}:{port}'.format(host=config["ollama_host"], port=config["ollama_port"])
ollama_client = Client(
  host=ollama_url,
)
print("⚙️ Set Ollama url to:", ollama_url)

# DOCUMENT READING
documents = read_text_files(config["data_dir"])
print("🔎Read ", len(documents)," files. Chunking and embedding...")
# for document in documents:
#     print("Filename: ", document["source"])

# CHUNK, EMBEDDINGS GENERATION AND STORAGE TO DB
for document in documents:
    text = document["text"]
    filename = document["source"]

    chunks = semantic_chunker(text,  config["chunk_size"], config["chunk_overlap"], "synonyms.csv")
    embeds = generate_embeddings(chunks, ollama_client, config["embedding_model"], config["batch_size"])
    data = [{
        "id": np.int64(int(hashlib.md5(f"{document['source']}_{i}".encode()).hexdigest()[:15], 16)),  # Convert to 64-bit int,
        "text": chunk,
        "source": document["source"],
        "vector": emb
    } for i, (chunk, emb) in enumerate(zip(chunks, embeds))]

    store_in_milvus(data, collection_name=config["collection_name"], milvus_path=config["milvus_path"])

    print(f"\n📁 File Completed: ", document["source"])
    print(f"📝 Total chunks: {len(chunks)}")
    print("──────────────────────────────")
    
print("✅Embeddings stored successfully.")
print("Milvus file: ",config["milvus_path"])
print("Milvus Collection name: ",config["collection_name"])