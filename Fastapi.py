from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
from pymilvus import MilvusClient
from utils import load_config

app = FastAPI()
config = load_config()

# Initialize clients
ollama_client = Client(host=f"http://{config['ollama_host']}:{config['ollama_port']}")
milvus_client = MilvusClient(config["milvus_path"])

class QueryRequest(BaseModel):
    question: str

def retrieve_context(query: str) -> Optional[str]:
    """Retrieve relevant context from Milvus"""
    try:
        embedding = ollama_client.embeddings(
            model=config["embedding_model"],
            prompt=query
        )["embedding"]
        
        results = milvus_client.search(
            collection_name=config["collection_name"],
            data=[embedding],
            limit=config["top_k"],
            output_fields=["text", "source"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        
        relevant_chunks = [
            f"SOURCE: {hit['entity']['source']}\n{hit['entity']['text']}"
            for hit in results[0]
            if hit["distance"] >= config["score_threshold"]
        ]
        
        return "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else None
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return None

@app.post("/ask")
async def ask_question(request: QueryRequest) -> str:
    """Single endpoint RAG query handler"""
    try:
        context = retrieve_context(request.question) or "No context found"
        
        response = ollama_client.chat(
            model=config["chat_model"],
            messages=[{
                'role': 'user',
                'content': f"Context: {context}\nQuestion: {request.question}"
            }],
            options={'temperature': 0.3}
        )
        
        return response['message']['content']
        
    except Exception as e:
        raise HTTPException(500, f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)