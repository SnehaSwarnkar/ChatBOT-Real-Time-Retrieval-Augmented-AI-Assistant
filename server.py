
import os
import base64
from io import BytesIO

import ollama

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader


from chat import ask_question, collection, LLM_MODEL


# ==================================================
# 1. CONFIGURATION
# ==================================================

APP_NAME = "Personal AI Assistant"

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "llava:7b"
)

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB per file

MAX_CONTEXT_CHARS = 30000

ALLOWED_PDF = {".pdf"}

ALLOWED_IMAGES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
}


# ==================================================
# 2. INITIALIZE FASTAPI
# ==================================================

app = FastAPI(
    title=APP_NAME,
    description="Local ChatGPT-style RAG and multimodal AI assistant",
    version="2.0.0"
)

# Use OpenAPI 3.0 for Swagger file-upload compatibility
app.openapi_version = "3.0.2"


# ==================================================
# 3. CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==================================================
# 4. REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=10000
    )


# ==================================================
# 5. PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(file_bytes, filename):

    try:

        pdf = PdfReader(BytesIO(file_bytes))

        extracted_pages = []

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                extracted_pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": text.strip()
                })

        return extracted_pages

    except Exception as error:

        raise ValueError(
            f"Could not read PDF {filename}: {error}"
        )


# ==================================================
# 6. IMAGE UNDERSTANDING
# ==================================================

def understand_image(file_bytes, filename):

    try:

        encoded_image = base64.b64encode(
            file_bytes
        ).decode("utf-8")

        response = ollama.chat(

            model=VISION_MODEL,

            messages=[
                {
                    "role": "user",

                    "content": (
                        "Analyze this image carefully. "
                        "Describe all visible objects, "
                        "text, diagrams, charts, labels, "
                        "and important details. "
                        "Do not invent information. "
                        "If something is unclear, mention it."
                    ),

                    "images": [encoded_image]
                }
            ],

            stream=False
        )

        return response.message.content

    except Exception as error:

        raise ValueError(
            f"Image processing failed for {filename}. "
            f"Check whether {VISION_MODEL} is installed. "
            f"Details: {error}"
        )


# ==================================================
# 7. MULTIMODAL FILE PROCESSING
# ==================================================

async def process_uploaded_files(files):

    document_context = []
    sources = []

    total_size = 0

    for uploaded_file in files:

        filename = os.path.basename(
            uploaded_file.filename or "unknown"
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        file_bytes = await uploaded_file.read()

        total_size += len(file_bytes)

        if len(file_bytes) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 15 MB limit."
            )

        if total_size > 30 * 1024 * 1024:

            raise HTTPException(
                status_code=413,
                detail="Combined upload size exceeds 30 MB."
            )

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail=f"{filename} is empty."
            )

        # ------------------------------------------
        # PDF PROCESSING
        # ------------------------------------------

        if extension in ALLOWED_PDF:

            try:

                pages = extract_pdf_text(
                    file_bytes,
                    filename
                )

                if not pages:

                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"No extractable text found in "
                            f"{filename}. Scanned PDFs need OCR."
                        )
                    )

                for page in pages:

                    document_context.append(
                        f"\n[File: {filename} | "
                        f"Page: {page['page']}]\n"
                        f"{page['text']}"
                    )

                    sources.append({
                        "source": filename,
                        "page": page["page"]
                    })

            except HTTPException:

                raise

            except Exception as error:

                raise HTTPException(
                    status_code=400,
                    detail=str(error)
                )

        # ------------------------------------------
        # IMAGE PROCESSING
        # ------------------------------------------

        elif extension in ALLOWED_IMAGES:

            description = understand_image(
                file_bytes,
                filename
            )

            document_context.append(
                f"\n[Image: {filename}]\n"
                f"{description}"
            )

            sources.append({
                "source": filename,
                "page": "Image"
            })

        else:

            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type: {extension}. "
                    "Upload PDF, PNG, JPG, JPEG or WEBP."
                )
            )

    return document_context, sources


# ==================================================
# 8. GENERATE ANSWER USING LOCAL LLM
# ==================================================

def generate_file_answer(question, document_context):

    context = "\n".join(document_context)

    context = context[:MAX_CONTEXT_CHARS]

    system_prompt = """
You are My AI, a helpful personal AI assistant.

Answer the user's questions using the supplied
document and image context.

Instructions:

1. Understand the provided context carefully.
2. Give clear and accurate answers.
3. Explain difficult concepts in simple language.
4. Use headings and bullet points when useful.
5. Do not invent facts or citations.
6. If the answer is not present in the context,
   clearly say that the provided files do not
   contain enough information.
7. Mention relevant page numbers when possible.
8. Be conversational and helpful.
"""

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",

                "content": (
                    f"DOCUMENT AND IMAGE CONTEXT:\n"
                    f"{context}\n\n"
                    f"USER QUESTION:\n{question}"
                )
            }
        ],

        stream=False
    )

    return response.message.content


# ==================================================
# 9. HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "running",
        "assistant": APP_NAME,
        "model": LLM_MODEL,
        "vision_model": VISION_MODEL,
        "documents_loaded": collection.count()
    }


# ==================================================
# 10. NORMAL CHAT ENDPOINT
# ==================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    question = request.message.strip()

    if not question:

        raise HTTPException(

            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        answer, metadata = ask_question(question)

        sources = []

        for meta in metadata:

            sources.append({
                "source": meta.get(
                    "source",
                    "Unknown"
                ),
                "page": meta.get(
                    "page",
                    meta.get("slide", "Unknown")
                )
            })

        return {
            "answer": answer,
            "sources": sources,
            "model": LLM_MODEL
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==================================================
# 11. FILE UPLOAD + CHAT ENDPOINT
# ==================================================

@app.post("/api/chat/upload")
async def chat_with_files(

    message: str = Form(...),

    files: list[UploadFile] = File(...)
):

    question = message.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    if not files:

        raise HTTPException(
            status_code=400,
            detail="Please upload at least one file."
        )

    try:

        document_context, sources = (
            await process_uploaded_files(files)
        )

        if not document_context:

            raise HTTPException(
                status_code=400,
                detail="No usable content was extracted."
            )

        answer = generate_file_answer(
            question,
            document_context
        )

        return {
            "answer": answer,
            "sources": sources,
            "model": LLM_MODEL,
            "files_processed": len(files)
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==================================================
# 12. ROOT ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to your Personal AI Assistant!",
        "documentation": "/docs",
        "health": "/api/health",
        "status": "online"
    }
