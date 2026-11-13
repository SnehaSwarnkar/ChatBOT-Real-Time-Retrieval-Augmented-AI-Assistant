
import os
import hashlib
import pymupdf
import chromadb
import ollama

from pptx import Presentation


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_FOLDER = "data"

EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 32

# Separate collection for the multi-format RAG pipeline
chroma_client = chromadb.PersistentClient(
    path="./chroma_db_ollama"
)

collection = chroma_client.get_or_create_collection(
    name="multi_format_documents",
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 2. EXTRACT TEXT FROM PDF
# ==========================================

def extract_pdf_pages(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text().strip()

        if text:

            pages.append({
                "text": text,
                "page": page_number,
                "source": os.path.basename(pdf_path),
                "file_type": "PDF"
            })

    doc.close()

    return pages


# ==========================================
# 3. EXTRACT TEXT FROM POWERPOINT
# ==========================================

def extract_pptx_slides(pptx_path):

    if not os.path.exists(pptx_path):
        raise FileNotFoundError(pptx_path)

    presentation = Presentation(pptx_path)

    slides = []

    for slide_number, slide in enumerate(
        presentation.slides, start=1
    ):

        slide_text = []

        for shape in slide.shapes:

            # Extract text boxes and headings
            if shape.has_text_frame:

                text = shape.text.strip()

                if text:
                    slide_text.append(text)

            # Extract tables
            if shape.has_table:

                for row in shape.table.rows:

                    row_text = []

                    for cell in row.cells:

                        if cell.text.strip():
                            row_text.append(cell.text.strip())

                    if row_text:
                        slide_text.append(" | ".join(row_text))

        text = "\n".join(slide_text)

        if text.strip():

            slides.append({
                "text": text,
                "page": slide_number,
                "source": os.path.basename(pptx_path),
                "file_type": "PPTX"
            })

    return slides


# ==========================================
# 4. CHUNK TEXT
# ==========================================

def chunk_text(text, chunk_size=CHUNK_SIZE,
               overlap=CHUNK_OVERLAP):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = words[i:i + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


# ==========================================
# 5. GENERATE EMBEDDINGS USING OLLAMA
# ==========================================

def generate_embeddings(chunks):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    embeddings = response.embeddings

    print(
        f"Generated {len(embeddings)} embeddings"
    )

    print(
        "Embedding dimensions:",
        len(embeddings[0])
    )

    return embeddings


# ==========================================
# 6. STORE CHUNKS IN CHROMADB
# ==========================================

def store_chunks(all_chunks, all_metadata, all_ids):

    if not all_chunks:
        print("No chunks available to store.")
        return

    total_chunks = len(all_chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        end = start + BATCH_SIZE

        batch_chunks = all_chunks[start:end]

        batch_metadata = all_metadata[start:end]

        batch_ids = all_ids[start:end]

        print(
            f"\nGenerating embeddings for "
            f"{start + 1} to {min(end, total_chunks)}"
        )

        embeddings = generate_embeddings(batch_chunks)

        collection.upsert(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=batch_metadata
        )

        print(
            f"Stored {min(end, total_chunks)} "
            f"of {total_chunks} chunks"
        )

    print("\nAll chunks stored successfully!")

    print(
        "Total records in collection:",
        collection.count()
    )


# ==========================================
# 7. PROCESS ALL PDF AND PPTX FILES
# ==========================================

def process_all_documents():

    all_chunks = []

    all_metadata = []

    all_ids = []

    # Find all supported documents
    document_files = sorted([

        f for f in os.listdir(DATA_FOLDER)

        if f.lower().endswith((".pdf", ".pptx"))

    ])

    if not document_files:

        print("No PDF or PPTX files found in data folder!")

        return

    print(
        f"\nFound {len(document_files)} documents."
    )

    total_pages = 0

    for document_file in document_files:

        document_path = os.path.join(
            DATA_FOLDER,
            document_file
        )

        print("\n--------------------------------")
        print("Processing:", document_file)
        print("--------------------------------")

        # Select extraction method
        if document_file.lower().endswith(".pdf"):

            pages = extract_pdf_pages(document_path)

        elif document_file.lower().endswith(".pptx"):

            pages = extract_pptx_slides(document_path)

        else:
            continue

        print("Extracted pages/slides:", len(pages))

        total_pages += len(pages)

        document_chunk_count = 0

        # Process every page or slide
        for page_data in pages:

            chunks = chunk_text(page_data["text"])

            for chunk_index, chunk in enumerate(chunks):

                # Create stable unique IDs
                unique_text = (
                    f"{page_data['source']}_"
                    f"{page_data['page']}_"
                    f"{chunk_index}"
                )

                document_hash = hashlib.sha1(
                    page_data["source"].encode("utf-8")
                ).hexdigest()[:12]

                chunk_id = (
                    f"{document_hash}_"
                    f"page_{page_data['page']}_"
                    f"chunk_{chunk_index}"
                )

                all_chunks.append(chunk)

                all_metadata.append({

                    "source": page_data["source"],

                    "page": page_data["page"],

                    "file_type": page_data["file_type"]

                })

                all_ids.append(chunk_id)

                document_chunk_count += 1

        print(
            "Chunks created:",
            document_chunk_count
        )

    # Generate and store all embeddings
    print("\n================================")
    print("STARTING EMBEDDING GENERATION")
    print("================================")

    print("Total documents:", len(document_files))

    print("Total pages/slides:", total_pages)

    print("Total chunks:", len(all_chunks))

    store_chunks(
        all_chunks,
        all_metadata,
        all_ids
    )

    print("\n================================")
    print("RAG INGESTION COMPLETED!")
    print("================================")

    print("Documents processed:", len(document_files))

    print("Pages/slides processed:", total_pages)

    print("Chunks processed:", len(all_chunks))

    print("Database records:", collection.count())


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("\nStarting Multi-Format RAG Chatbot...")

    print("Embedding model:", EMBEDDING_MODEL)

    print("Storage: ChromaDB")

    print("Supported formats: PDF, PPTX")

    process_all_documents()
