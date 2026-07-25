from pathlib import Path

from pypdf import PdfReader


# Main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder containing annual reports and production reports
DOCUMENTS_FOLDER = PROJECT_ROOT / "documents"


def find_pdf_files() -> list[Path]:
    """
    Find every PDF file inside the documents folder.
    """

    if not DOCUMENTS_FOLDER.exists():
        raise FileNotFoundError(
            f"Documents folder was not found: {DOCUMENTS_FOLDER}"
        )

    pdf_files = sorted(DOCUMENTS_FOLDER.rglob("*.pdf"))

    return pdf_files


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        extracted_pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    return extracted_pages

def clean_text(text: str) -> str:
    """
    Remove unnecessary spaces and line breaks.
    """

    cleaned_text = " ".join(text.split())

    return cleaned_text

def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """
    Divide cleaned text into overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def main() -> None:
    """
    Test PDF reading, cleaning and chunking.
    """

    pdf_files = find_pdf_files()

    print(f"PDF files found: {len(pdf_files)}")

    if not pdf_files:
        print("No PDF files were found.")
        return

    test_pdf = pdf_files[0]

    print(f"\nTesting PDF: {test_pdf.name}")

    pages = extract_pdf_pages(test_pdf)

    print(f"Number of pages: {len(pages)}")

    first_text_page = None

    for page_data in pages:
        if page_data["text"]:
            first_text_page = page_data
            break

    if first_text_page is None:
        print("No readable text was found.")
        return

    print(
        f"First readable page: "
        f"{first_text_page['page']}"
    )

    cleaned_text = clean_text(
        first_text_page["text"]
    )

    print(
        f"Cleaned text length: "
        f"{len(cleaned_text)} characters"
    )

    chunks = split_text_into_chunks(
        cleaned_text,
        chunk_size=800,
        overlap=120,
    )

    print(f"Chunks created: {len(chunks)}")

    for number, chunk in enumerate(
        chunks[:2],
        start=1,
    ):
        print(f"\nCHUNK {number}:\n")
        print(chunk)
        
        
if __name__ == "__main__":
    main()