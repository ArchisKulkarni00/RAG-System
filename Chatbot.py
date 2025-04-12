from typing import Dict, Any
from utils import load_config
from ollama import Client

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
    
    {context if context else 'No specific context provided'}

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
        context = None  # Replace with your retrieval logic
        
        # 3. Get LLM response
        response = query_ollama(query, context)
        
        # 4. Display response
        print("\n💡 Response:")
        print(response)

# Example usage:
if __name__ == "__main__":
    # Sample config - replace with your actual config
    test_rag_pipeline()