import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json
from langchain.schema import Document
import os
import pytesseract

# Absolute path to your local tessdata
os.environ['TESSDATA_PREFIX'] = os.path.abspath("./tessdata")

# Point to your tesseract.exe directly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def load_arabic_pdf(pdf_path, lang="ara", batch_size=40, cache_dir="config_files/ktebjson"):
    os.makedirs(cache_dir, exist_ok=True)

    # pdf_name = os.path.basename(pdf_path)
    # cache_file = os.path.join(cache_dir, pdf_name + ".json")
    cache_file="config_files\ktebjson\Book.json"
    if os.path.exists(cache_file):
        print(f"Loading cached OCR data from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]

    documents = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size, total_pages)
            print(f"Processing pages {start+1} to {end}...")

            for i in range(start, end):
                page = doc[i]
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang=lang)

                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_path,
                            "page": i,
                            "total_pages": total_pages,
                            "page_label": str(i + 1)
                        }
                    )
                )

        doc.close()

        # Save to cache as JSON-serializable format
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
                f,
                ensure_ascii=False,
                indent=2
            )

        return documents

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []
