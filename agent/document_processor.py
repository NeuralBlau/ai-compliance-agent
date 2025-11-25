import pytesseract
from pdf2image import convert_from_path
import os
import boto3
from PIL import Image

# --- KONSTANTEN ---
S3_BUCKET_NAME = "neuralblau-ai-compliance-agent-documents"
S3_PREFIX = "raw/"
LOCAL_RAW_DIR = "data/raw"
ALLOWED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg']

# --- S3 DOWNLOAD FUNKTION (unverändert) ---
def download_documents_from_s3(bucket: str, prefix: str) -> list[str]:
    """Lädt alle Dokumente unter einem S3-Präfix herunter und gibt die lokalen Pfade zurück."""
    s3 = boto3.client('s3')
    local_paths = []
    
    print(f"Suche Dokumente in s3://{bucket}/{prefix}...")
    
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as e:
        print(f"Konnte S3-Bucket nicht abfragen. AWS-CLI/Konfiguration korrekt? Fehler: {e}")
        return []

    if 'Contents' not in response:
        print("Keine Dokumente im S3-Bucket unter diesem Präfix gefunden.")
        return []

    for item in response['Contents']:
        s3_key = item['Key']
        if s3_key == prefix or s3_key.endswith('/'):
            continue
            
        filename = os.path.basename(s3_key)
        
        # NEU: Überprüfung der Erweiterung
        if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            print(f"--> WARNUNG: {filename} übersprungen (unerlaubtes Format).")
            continue
            
        local_path = os.path.join(LOCAL_RAW_DIR, filename)
        
        try:
            s3.download_file(bucket, s3_key, local_path)
            print(f"--> Erfolgreich heruntergeladen: {filename}")
            local_paths.append(local_path)
        except Exception as e:
            print(f"Fehler beim Download von {s3_key}: {e}")

    return local_paths

# --- 1. OCR FÜR PDF (Multi-Page) ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrahiert Text aus einem PDF (unterstützt gescannte Dokumente via OCR)."""
    full_text = []
    
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Fehler bei Konvertierung (Poppler-Fehler?): {e}")
        return "[FEHLER: PDF-Konvertierung]"

    for i, image in enumerate(images):
        # Vorverarbeitung: Graustufen für bessere OCR-Genauigkeit
        image = image.convert('L')
        try:
            text = pytesseract.image_to_string(image, lang='deu')
            full_text.append(f"--- Seite {i+1} ---\n{text}")
        except Exception as e:
            print(f"Tesseract Fehler auf Seite {i+1}: {e}")
            full_text.append(f"--- Seite {i+1} ---\n[OCR FEHLER]")

    return "\n\n".join(full_text)

# --- 2. OCR FÜR BILDER (Single-Image) ---
def extract_text_from_image(image_path: str) -> str:
    """Extrahiert Text aus einer einzelnen Bilddatei (PNG, JPG)."""
    try:
        image = Image.open(image_path)
        # Vorverarbeitung: Graustufen
        image = image.convert('L')
        text = pytesseract.image_to_string(image, lang='deu')
        return text
    except Exception as e:
        print(f"Fehler bei Bild-OCR von {image_path}: {e}")
        return "[FEHLER: Bild-OCR]"


# --- ZENTRALE VERARBEITUNGSLOGIK ---
def process_document(local_path: str) -> str:
    """Wählt die passende OCR-Funktion basierend auf der Dateierweiterung."""
    ext = os.path.splitext(local_path)[1].lower()
    
    if ext == '.pdf':
        print("  -> Verarbeite als Multi-Page PDF.")
        return extract_text_from_pdf(local_path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        print("  -> Verarbeite als Single-Image.")
        return extract_text_from_image(local_path)
    else:
        return "[FEHLER: Unerlaubtes oder unbekanntes Dateiformat]"


# --- Hauptausführung anpassen ---
if __name__ == "__main__":
    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Starte Dokumentenverarbeitung...")

    # 1. Dokumente von S3 herunterladen
    document_paths = download_documents_from_s3(S3_BUCKET_NAME, S3_PREFIX)

    if not document_paths:
        print("\nKeine Dateien zum Verarbeiten. Bitte prüfen Sie S3 Bucket/Prefix/AWS Config.")
    else:
        # 2. Iteriere über alle heruntergeladenen Dateien und führe OCR aus
        for pdf_path in document_paths:
            print(f"\nVerarbeite: {pdf_path}")
            extracted_content = process_document(pdf_path) # NEU: Nutzt zentrale Logik
            
            # 3. Speichere das Ergebnis
            filename = os.path.basename(pdf_path)
            # Ersetze die Dateiendung durch .txt
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join("data/processed", output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_content)
            print(f"Erfolgreich OCR-Text gespeichert unter: {output_path}")