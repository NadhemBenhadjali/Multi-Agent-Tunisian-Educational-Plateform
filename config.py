import os
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Environment & external service credentials (from original script) ---
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'
os.environ["GEMINI_API_KEY"] = "AIzaSyA6qNq5Gc08rtJviKT_A3kd2p0o82o2KNo"
PDF_PATH = "config_files/Book.pdf"  # ← change to your PDF if needed
# Neo4j connection
URI = "neo4j+s://d24579bb.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "C7ebpFAjR9JcM1QbLPJHy5R91gwzaQOUJoBvoUGhfWw"  # ← change to your Neo4j password
# --- Arabic font & images configuration (from original script) ---
ARABIC_FONT_PATH = "config_files/NotoNaskhArabic-Regular.ttf"     # ← change if needed
ARABIC_FONT_NAME = "NotoArabic"
IMG_DIR = "config_files/book_images"     # adjust if your folder differs
MAX_IMG_W = 180                           # pixel width allowed on page
MAX_IMG_H = 140                           # pixel height allowed on page

# Markdown image tag regex
MD_IMG = re.compile(r'!\[(.*?)\]\((.*?)\)')   # ![alt](path)

# Register the Arabic font
pdfmetrics.registerFont(TTFont(ARABIC_FONT_NAME, ARABIC_FONT_PATH))
