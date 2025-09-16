from pathlib import Path
from kg import Neo4jKG
from cli import run_cli
from config import URI, USER, PASSWORD
def main():
    sample_pdf = Path("config_files/الإيقاظ العلمي - السنة الرابعة من التعليم الأساسي (1).pdf")
    neo_kg = Neo4jKG(URI, USER, PASSWORD)
    if sample_pdf.exists():
        run_cli(sample_pdf, neo_kg)
    else:
        print("PDF not found, please adjust sample_pdf path first.")

if __name__ == "__main__":
    main()