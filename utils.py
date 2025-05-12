import os
from typing import List, Dict, Any
from tqdm import tqdm
import yaml
from ollama import Client
import csv
from collections import defaultdict
import spacy
from pymilvus import MilvusClient

# Load Spacy's English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy English model 'en_core_web_sm' not found. Please install it first.")
    exit(1)

PREFIX_SEARCH_DOC = "search_document:"
PREFIX_SEARCH_QUERY = "search_query:"


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

def semantic_chunker(text: str, chunk_size: int = 400, overlap: int = 40, synonyms_file=None) -> List[str]:
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
                current_chunk_text = ' '.join(current_chunk)
                current_chunk_text = enhance_text_with_metadata(current_chunk_text)
                chunks.append(current_chunk_text)
                current_chunk = current_chunk[-overlap:]  # Apply overlap
                current_length = len(current_chunk)
            
            # Add as much of the paragraph as possible
            while para_words:
                remaining = chunk_size - current_length
                current_chunk.extend(para_words[:remaining])
                current_length += remaining
                para_words = para_words[remaining:]
                
                if current_length >= chunk_size:
                    current_chunk_text = ' '.join(current_chunk)
                    current_chunk_text = enhance_text_with_metadata(current_chunk_text)
                    chunks.append(current_chunk_text)
                    current_chunk = current_chunk[-overlap:]
                    current_length = len(current_chunk)
        
    
    # Add remaining words
    if current_chunk: 
        current_chunk_text = ' '.join(current_chunk)
        current_chunk_text = enhance_text_with_metadata(current_chunk_text)
        current_chunk_text = PREFIX_SEARCH_DOC+current_chunk_text
        chunks.append(current_chunk_text)
    
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

def retrieve_context(
    query: str,
    ollama_client,
    milvus_path,
    embedding_model,
    collection_name,
    top_k,
    score_threshold
):
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
        client = MilvusClient(milvus_path)

        query = PREFIX_SEARCH_QUERY + query
        
        # 2. Generate query embedding
        embedding = ollama_client.embeddings(
            model=embedding_model,
            prompt=query
        )["embedding"]
        
        # 3. Search Milvus
        results = client.search(
            collection_name=collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["text", "source"],  # Return these fields
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        
        # 4. Filter and format results
        relevant_chunks = []
        for hit in results[0]:
            if hit["distance"] >= score_threshold:  # Higher score = more relevant
                source = hit["entity"]["source"]
                text = hit["entity"]["text"]
                
                # Remove the [Keywords: ...] header if it exists
                if text.startswith("[Keywords:") and "\n\n" in text:
                    # Find the end of the keywords header (after the double newline)
                    header_end = text.find("\n\n") + 2
                    text = text[header_end:]  # Keep everything after the header
                    
                relevant_chunks.append(f"SOURCE: {source}\n{text}")
        
        return "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else None
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return None

def load_synonyms_from_csv(csv_path="synonyms.csv"):
    """
    Load synonyms from CSV file with columns: Original,Synonyms
    Returns dictionary of {original: [synonyms]} or None if file doesn't exist
    """
    if not os.path.exists(csv_path):
        return None
        
    synonyms = defaultdict(list)
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row['Original'].strip().lower()
            if row['Synonyms']:
                for syn in row['Synonyms'].split(','):
                    synonyms[original].append(syn.strip().lower())
    return dict(synonyms) if synonyms else None

def extract_important_terms(text):
    """Extract nouns, proper nouns, and noun phrases from text."""
    doc = nlp(text.lower())
    terms = set()
    
    # Extract nouns and proper nouns
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
            terms.add(token.lemma_)
    
    # Extract noun chunks (e.g., "entrance exam")
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 3:
            normalized = " ".join([t.lemma_ for t in chunk])
            terms.add(normalized)
    
    return terms

def augment_with_synonyms(terms, synonym_dict=None):
    """
    Expand terms with synonyms from provided dictionary.
    If no dictionary provided, returns original terms.
    """
    if not synonym_dict:
        return terms
        
    augmented = set()
    for term in terms:
        augmented.add(term)
        # Check for exact matches
        if term in synonym_dict:
            augmented.update(synonym_dict[term])
        # Check for partial matches in multi-word terms
        for key, syns in synonym_dict.items():
            if key in term and key != term:
                augmented.update(syns)
    return augmented

def enhance_text_with_metadata(text, synonym_file=None):
    """
    Enhance text by prepending extracted (and optionally synonym-augmented) metadata.
    If synonym_file doesn't exist, skips synonym augmentation.
    """
    # Load synonyms if file exists
    synonym_dict = None
    if synonym_file:
        synonym_dict = load_synonyms_from_csv(synonym_file)
        if synonym_dict is None:
            print(f"Note: Synonym file '{synonym_file}' not found - skipping synonym augmentation")
    
    # Step 1: Extract important terms
    terms = extract_important_terms(text)
    
    # Step 2: Optionally augment with synonyms
    augmented_terms = augment_with_synonyms(terms, synonym_dict)
    
    # Step 3: Format metadata header
    metadata_header = "[Keywords: " + ", ".join(sorted(augmented_terms)) + "]\n\n"
    
    # Step 4: Combine with original text
    enhanced_text = metadata_header + text
    
    return enhanced_text
