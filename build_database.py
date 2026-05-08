import os
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer  # Fallback: bge-m3 本地模型
import time
from tqdm import tqdm
import sys
import io
import requests

# --- Configuration ---
DB_PATH = "./database"  # Your Database data path
COLLECTION_NAME = "my_collection"
LOCAL_EMBEDDING_MODEL = 'BAAI/bge-m3'
LOCAL_ENCODING_BATCH_SIZE = 32
CHUNK_SIZE = 500   # Adjust based on real case
EMBEDDING_API_BATCH_SIZE = 32
UPLOAD_BATCH_SIZE = 500  # Adjust based on your system's memory

# --- SiliconFlow Embedding API Config ---
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
MAX_RETRIES = 3


# ==============================================================================
# SiliconFlow Embedding API
# ==============================================================================

def _siliconflow_headers():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        return None
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def get_embeddings_via_api(texts, model=None, batch_size=EMBEDDING_API_BATCH_SIZE):
    """
    调用 SiliconFlow Embedding API 将文本列表转为向量。
    单次最多 batch_size 条输入，超出自动分批。
    返回与 texts 等长的向量列表。
    """
    if model is None:
        model = os.getenv("SILICONFLOW_EMBEDDING_MODEL", SILICONFLOW_EMBEDDING_MODEL)

    headers = _siliconflow_headers()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings (SiliconFlow)"):
        batch = texts[i:i + batch_size]
        payload = {
            "model": model,
            "input": batch,
            "encoding_format": "float",
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=60)
                result = resp.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\n  Embedding API request failed, retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"Embedding API request failed ({MAX_RETRIES} retries): {e}")

        if resp.status_code != 200 or "data" not in result:
            raise RuntimeError(f"SiliconFlow Embedding API error: HTTP {resp.status_code}, {result}")

        sorted_data = sorted(result["data"], key=lambda x: x["index"])
        for item in sorted_data:
            all_embeddings.append(item["embedding"])

    print(f"Embedding generation complete. Total vectors: {len(all_embeddings)}")
    return all_embeddings


# --- Text Extraction Functions ---

def extract_text_from_pdf(pdf_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a PDF and splits it into chunks."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        print(f"Warning: Could not process PDF {pdf_path}. Error: {e}")
        return []


def extract_text_from_txt(txt_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a TXT file and splits it into chunks."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        print(f"Warning: Could not process TXT {txt_path}. Error: {e}")
        return []


def extract_text_from_md(md_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a MD file and splits it into chunks."""
    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        print(f"Warning: Could not process MD {md_path}. Error: {e}")
        return []


# --- Main Database Build Logic ---

def build_database():
    """
    Initializes ChromaDB, processes source documents, generates embeddings,
    and populates the database.
    """
    print("--- Starting Database Build Process ---")

    # 1. Connect to ChromaDB
    print(f"Connecting to database at: {DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Safety Check: Ask before overwriting existing data
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() > 0:
            print(f"\n⚠️  Warning: Collection '{COLLECTION_NAME}' already contains {collection.count()} documents.")
            choice = input("Do you want to delete existing data and rebuild? (y/N): ").lower()
            if choice == 'y':
                print(f"Deleting existing collection '{COLLECTION_NAME}'...")
                chroma_client.delete_collection(name=COLLECTION_NAME)
                collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
                print("Collection deleted. Proceeding with rebuild.")
            else:
                print("Aborting build process. Database remains unchanged.")
                return
    else:
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # 3. Process all PDF, TXT, and MD files in the directory
    print(f"\nProcessing files from directory: {DB_PATH}")
    all_text_chunks = []
    all_document_ids = []

    # Get a list of files to process (now includes .md)
    files_to_process = [f for f in os.listdir(DB_PATH) if f.endswith('.pdf') or f.endswith('.txt') or f.endswith('.md')]
    if not files_to_process:
        print(f"No .pdf, .txt, or .md files found in {DB_PATH}. Exiting.")
        return

    # Process files with a tqdm progress bar
    for filename in tqdm(files_to_process, desc="Processing Files"):
        file_path = os.path.join(DB_PATH, filename)
        chunks = []
        if filename.endswith('.pdf'):
            chunks = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            chunks = extract_text_from_txt(file_path)
        elif filename.endswith('.md'):
            chunks = extract_text_from_md(file_path)

        start_id = len(all_document_ids)
        all_text_chunks.extend(chunks)
        all_document_ids.extend([f"{filename}_{start_id + i}" for i in range(len(chunks))])

    if not all_text_chunks:
        print("\nNo text chunks were extracted. Make sure your files are not empty or corrupted.")
        return

    print(f"\n✅ Total text chunks created: {len(all_text_chunks)}")

    # 4. Generate Embeddings: SiliconFlow API 优先，失败 fallback 本地 bge-m3
    all_embeddings = None
    try:
        headers = _siliconflow_headers()
        if headers is not None:
            print(f"\nGenerating embeddings via SiliconFlow API (model: {SILICONFLOW_EMBEDDING_MODEL})...")
            all_embeddings = get_embeddings_via_api(all_text_chunks)
        else:
            print("SILICONFLOW_API_KEY not set, using local model.")
    except Exception as e:
        print(f"SiliconFlow API failed ({e}), falling back to local model.")

    if all_embeddings is None:
        print(f"\nLoading local embedding model: {LOCAL_EMBEDDING_MODEL}...")
        start_time = time.time()
        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

        print("Generating embeddings for all text chunks...")
        all_embeddings = []
        for i in tqdm(range(0, len(all_text_chunks), LOCAL_ENCODING_BATCH_SIZE), desc="Generating Embeddings"):
            batch_chunks = all_text_chunks[i:i + LOCAL_ENCODING_BATCH_SIZE]
            batch_embeddings = model.encode(batch_chunks, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        print(f"Embedding generation complete. Total vectors: {len(all_embeddings)}")

    # 5. Upload data to ChromaDB in batches with Progress Bar
    print("\nUploading documents and embeddings to ChromaDB...")

    for i in tqdm(range(0, len(all_text_chunks), UPLOAD_BATCH_SIZE), desc="Uploading to ChromaDB"):
        batch_embeddings = all_embeddings[i:i + UPLOAD_BATCH_SIZE]
        # API 返回 list，本地模型返回 ndarray — 统一为 list
        if batch_embeddings and hasattr(batch_embeddings[0], 'tolist'):
            batch_embeddings = [e.tolist() for e in batch_embeddings]
        collection.upsert(
            documents=all_text_chunks[i:i + UPLOAD_BATCH_SIZE],
            ids=all_document_ids[i:i + UPLOAD_BATCH_SIZE],
            embeddings=batch_embeddings
        )

    print("\n--- ✅ Database Build Complete! ---")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")


if __name__ == "__main__":
    # Ensure stdout supports UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            print("Stdout encoding set to UTF-8")
        except Exception as e:
            print(f"Warning: Could not set stdout encoding to UTF-8. Error: {e}")

    build_database()

