from typing import Dict, Any
from utils import load_config
from ollama import Client
from utils import retrieve_context
import requests
import json

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
    sys_prompt = config["system_prompt"]

    prompt = f"""
    Context: {context}
    Question: {query}
    """
        
    jsonPayload = {
                'model': config["chat_model"],
                'prompt': prompt,
                'system': sys_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1
                }
            }

    try:
        response = requests.post(
            url=ollama_url + '/api/generate',
            json= jsonPayload,
        )
        # print(response.json())
        return response.json()['response']
    except Exception as e:
        print(f"🚨 Error querying Ollama: {e}")
        return "Sorry, I encountered an error"


def test_rag_pipeline(print_context=False):
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
        context = retrieve_context(query,
                                   ollama_client,
                                   config["milvus_path"],
                                   config["embedding_model"],
                                   config["collection_name"],
                                   config["top_k"],
                                   config["score_threshold"])
        if not context:
            context = 'No specific context provided'

        if print_context:
            print("\n ⚙️ Context:  ", context)
        
        # 3. Get LLM response
        response = query_ollama(query, context)
        
        # 4. Display response
        print("\n💡 Response:")
        print(response)

# Example usage:
if __name__ == "__main__":
    # test entire application
    test_rag_pipeline(False)

    # print the context being reterived for some query
    # print(retrieve_context("What is the procedure for summer internship selection."))