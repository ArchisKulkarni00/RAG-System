import os
from typing import List, Dict, Any
from tqdm import tqdm
import yaml
from ollama import Client


def load_config() -> Dict[str, Any]:
    """Load configuration with fallback hierarchy:
    1. First try local_config.yml (for environment-specific overrides)
    2. Then try config.yml (main configuration)
    3. Finally use hardcoded defaults
    """
    # Default configuration (ensure all required keys exist)

    config_files_to_try = [
        "local_config.yml",  # Highest priority
        "config.yml"        # Secondary priority
    ]

    final_config = {}
    
    for config_file in config_files_to_try:
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                final_config.update(file_config)
                print(f"Loaded configuration from {config_file}")
                break  # Stop at first successfully loaded file
                
        except FileNotFoundError:
            continue  # Try next config file
        except yaml.YAMLError as e:
            print(f"Error parsing {config_file}: {e}")
            continue
    else:
        print("No config files found, using defaults")
    
    return final_config


def read_text_files(folder_path: str) -> List[Dict[str, str]]:
    """Read all text files from a directory with error handling.
    
    Args:
        folder_path: Path to directory containing text files
        
    Returns:
        List of dictionaries with 'text' and 'source' keys
    """
    documents = []
    
    for filename in tqdm(os.listdir(folder_path), desc="Reading files"):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:  # Skip empty files
                        documents.append({
                            "text": text,
                            "source": filename
                        })
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read().strip()
                        documents.append({
                            "text": text,
                            "source": filename
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
    
    return documents

def semantic_chunker(text: str, chunk_size: int = 400, overlap: int = 40) -> List[str]:
    """Enhanced chunking with paragraph awareness.
    
    Args:
        text: Input text to chunk
        chunk_size: Target word count per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_word_count = len(para_words)
        
        # Case 1: Paragraph fits entirely in current chunk
        if current_length + para_word_count <= chunk_size:
            current_chunk.extend(para_words)
            current_length += para_word_count
        else:
            # Case 2: Paragraph needs to be split
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap:]  # Apply overlap
                current_length = len(current_chunk)
            
            # Add as much of the paragraph as possible
            while para_words:
                remaining = chunk_size - current_length
                current_chunk.extend(para_words[:remaining])
                current_length += remaining
                para_words = para_words[remaining:]
                
                if current_length >= chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = current_chunk[-overlap:]
                    current_length = len(current_chunk)
    
    # Add remaining words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_embeddings(
    texts: List[str],
    ollama_client: Client,
    model: str = "nomic-embed-text",
    batch_size: int = 32
) -> List[List[float]]:
    """Generate embeddings for a list of texts using Ollama.
    
    Args:
        texts: List of text strings to embed
        model: Name of embedding model to use
        batch_size: Number of texts to process at once
        
    Returns:
        List of embedding vectors
    """
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        batch_embeddings = []
        
        # Process each text individually in the batch
        for text in batch:
            response = ollama_client.embeddings(
                model=model,
                prompt=text  # Single string input
            )
            batch_embeddings.append(response["embedding"])
        
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def store_in_milvus(
    data: List[Dict],
    collection_name: str,
    milvus_path: str = "./milvus_data",
    dimension: int = 768,
    metric_type: str = "COSINE"
) -> None:
    """Store documents with embeddings in Milvus.
    
    Args:
        data: List of dictionaries containing:
            - id: Unique identifier
            - text: Original text
            - source: Source document
            - embedding: Vector embedding
        collection_name: Name of Milvus collection
        milvus_path: Path to Milvus data storage
        dimension: Dimension of embeddings
        metric_type: Similarity metric type
    """
    from pymilvus import MilvusClient
    
    client = MilvusClient(milvus_path)
    
    # Create collection if needed
    if collection_name not in client.list_collections():
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type
        )
    
    # Insert data in batches
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size), desc="Storing in Milvus"):
        batch = data[i:i + batch_size]
        client.insert(
            collection_name=collection_name,
            data=batch
        )

def inspect_chunks_for_file(file_path: str, chunk_size: int = 400, overlap: int = 40) -> None:
    """Debug function to view chunks for a single file.
    
    Args:
        file_path: Path to text file
        chunk_size: Target word count per chunk
        overlap: Number of overlapping words
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    chunks = semantic_chunker(text, chunk_size, overlap)
    
    print(f"\n📁 File: {os.path.basename(file_path)}")
    print(f"📝 Total chunks: {len(chunks)}")
    print("──────────────────────────────")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n🔹 Chunk {i} ({len(chunk.split())} words, {len(chunk)} chars)")
        print("─" * 40)
        print(chunk)
        print("─" * 40)