import os
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Environment & external service credentials (from original script) ---
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'
os.environ["GEMINI_API_KEY"] = "AIzaSyCMXmtbs6fu3YuLhtbUwS4wM0t0_Izc4xQ"
os.environ["CHROMA_GOOGLE_GENAI_API_KEY"] = os.environ["GEMINI_API_KEY"]
os.environ["QDRANT_URL"]="https://07cc33cb-f09d-4add-b07f-8440c6bbdb54.us-west-2-0.aws.cloud.qdrant.io:6333"
os.environ["QDRANT_API_KEY"]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bbIl5bU8oQisaPH4D0TMBr4zz4mkuejR6Zp37izO-N4"
PDF_PATH = "config_files/Book.pdf"  # ← change to your PDF if needed
# Neo4j connection
URI = "neo4j+s://599c903b.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "pNpqkMHUrcDyiBsD9chZW2AHYpg3xf7jGzMPlPtbBOc" 
ARABIC_FONT_PATH = "config_files/NotoNaskhArabic-Regular.ttf"     
ARABIC_FONT_NAME = "NotoArabic"
IMG_DIR = "config_files/book_images"     # adjust if your folder differs
MAX_IMG_W = 180                           # pixel width allowed on page
MAX_IMG_H = 140                           # pixel height allowed on page

# Markdown image tag regex
MD_IMG = re.compile(r'!\[(.*?)\]\((.*?)\)')   # ![alt](path)

# Register the Arabic font
pdfmetrics.registerFont(TTFont(ARABIC_FONT_NAME, ARABIC_FONT_PATH))
embedder_cfg = {
    "provider": "google-generativeai",    
    "config": {
        "api_key": "AIzaSyCMXmtbs6fu3YuLhtbUwS4wM0t0_Izc4xQ",
        "model_name": "text-embedding-004",   
    },
}
