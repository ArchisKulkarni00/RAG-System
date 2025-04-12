from typing import Dict, Any, Optional
from utils import load_config
from ollama import Client
from pymilvus import MilvusClient

# READ THE CONFIG FILE
config = load_config()

# SETUP OLLAMA CLIENT
ollama_url = 'http://{host}:{port}'.format(host=config["ollama_host"], port=config["ollama_port"])
ollama_client = Client(
  host=ollama_url,
)
print("⚙️ Set Ollama url to:", ollama_url)


def get_user_input() -> str:
    """Get and validate user query input"""
    while True:
        query = input("\n🧠 Ask a question (or 'quit' to exit): ").strip()
        if query.lower() in ('exit', 'quit', 'q'):
            return None
        elif query:
            return query
        print("⚠️ Please enter a valid question")

def query_ollama(
    query: str,
    context: str = None,
) -> str:
    """
    Query Ollama with RAG context
    Args:
        query: User question
        config: Configuration dict
        context: Retrieved context (None for direct QA)
    Returns:
        Generated response
    """
    prompt = f"""Answer the question based only on the following context:
    
    {context}

    Question: {query}
    
    Answer clearly and concisely."""
    
    try:
        response = ollama_client.chat(
            model=config["chat_model"],
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.3  # More deterministic
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"🚨 Error querying Ollama: {e}")
        return "Sorry, I encountered an error"

def retrieve_context(
    query: str
) -> Optional[str]:
    """
    Retrieve relevant context from Milvus based on user query.
    
    Args:
        query: User's question/search query
        config: Configuration dictionary containing:
            - milvus_path: Path to Milvus data
            - collection_name: Name of your collection
            - embedding_model: Name of embedding model
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity score (0-1)
        
    Returns:
        Combined relevant context as string or None if no results
    """
    try:
        # 1. Initialize Milvus client
        client = MilvusClient(config["milvus_path"])
        
        # 2. Generate query embedding
        embedding = ollama_client.embeddings(
            model=config["embedding_model"],
            prompt=query
        )["embedding"]
        
        # 3. Search Milvus
        results = client.search(
            collection_name=config["collection_name"],
            data=[embedding],
            limit=config["top_k"],
            output_fields=["text", "source"],  # Return these fields
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        
        # 4. Filter and format results
        relevant_chunks = []
        for hit in results[0]:
            if hit["distance"] >= config["score_threshold"]:  # Higher score = more relevant
                source = hit["entity"]["source"]
                text = hit["entity"]["text"]
                relevant_chunks.append(f"SOURCE: {source}\n{text}")
        
        return "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else None
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return None


def test_rag_pipeline():
    """Basic chat interface for testing RAG"""
    print("🔍 RAG Testing Interface")
    print(f"Model: {config['chat_model']}")
    print("Type 'quit' to exit\n")
    
    while True:
        # 1. Get user input
        query = get_user_input()
        if query is None:
            break
        
        # 2. (Future) Retrieve context from Milvus here
        context = retrieve_context(query)  # Replace with your retrieval logic
        if not context:
            context = 'No specific context provided'
        
        # 3. Get LLM response
        response = query_ollama(query, context)
        
        # 4. Display response
        print("\n💡 Response:")
        print(response)

# Example usage:
if __name__ == "__main__":
    # test entire application
    test_rag_pipeline()

    # print the context being reterived for some query
    # print(retrieve_context("What is the procedure for summer internship selection."))