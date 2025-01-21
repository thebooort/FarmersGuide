import os
import PyPDF2
from tqdm import tqdm

def extract_text_from_pdfs(folder_path="C:/Users/share/Documents/FarmerGuide/FarmersGuide/database/papers"):
    """Extracts text from all PDFs in the specified folder."""
    
    # Ensure the directory exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: {folder_path}. Make sure the path is correct!")

    # Get all PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDF files found in {folder_path}. Please add PDFs!")

    documents = []
    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        pdf_path = os.path.join(folder_path, pdf_file)
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        documents.append({"text": text, "source": pdf_file})

    return documents

if __name__ == "__main__":
    docs = extract_text_from_pdfs()
    print(f"✅ Extracted {len(docs)} PDFs")
