from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
from pymilvus import MilvusClient
from utils import load_config, retrieve_context

app = FastAPI()
config = load_config()

# Initialize clients
ollama_client = Client(host=f"http://{config['ollama_host']}:{config['ollama_port']}")
milvus_client = MilvusClient(config["milvus_path"])

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest) -> str:
    """Single endpoint RAG query handler"""
    try:
        context = retrieve_context(request.question,
                                   ollama_client,
                                   config["milvus_path"],
                                   config["embedding_model"],
                                   config["collection_name"],
                                   config["top_k"],
                                   config["score_threshold"]) or "No context found"
        
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