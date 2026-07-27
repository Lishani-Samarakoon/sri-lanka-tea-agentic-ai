from pathlib import Path

from pypdf import PdfReader


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

# build_index.py is located inside:
# sri-lanka-tea-agent/src/rag/build_index.py
#
# parents[2] moves back to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder containing:
# documents/annual_reports/
# documents/production_reports/
DOCUMENTS_FOLDER = PROJECT_ROOT / "documents"


# ---------------------------------------------------------
# CHUNKING SETTINGS
# ---------------------------------------------------------

# Maximum number of characters in one chunk.
CHUNK_SIZE = 800

# Number of characters repeated between two chunks.
CHUNK_OVERLAP = 120


# ---------------------------------------------------------
# FIND PDF FILES
# ---------------------------------------------------------

def find_pdf_files() -> list[Path]:
    """
    Find every PDF file inside the documents folder.
    """

    if not DOCUMENTS_FOLDER.exists():
        raise FileNotFoundError(
            f"Documents folder was not found: {DOCUMENTS_FOLDER}"
        )

    pdf_files = sorted(
        DOCUMENTS_FOLDER.rglob("*.pdf")
    )

    return pdf_files


# ---------------------------------------------------------
# EXTRACT TEXT FROM PDF PAGES
# ---------------------------------------------------------

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from every page of one PDF file.

    Each returned dictionary contains:
    - page number
    - extracted text
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file was not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    extracted_pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        extracted_pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    return extracted_pages


# ---------------------------------------------------------
# CLEAN EXTRACTED TEXT
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove repeated spaces, tabs and line breaks.
    """

    if not isinstance(text, str):
        raise TypeError(
            "The text value must be a string."
        )

    cleaned_text = " ".join(
        text.split()
    )

    return cleaned_text


# ---------------------------------------------------------
# SPLIT TEXT INTO CHUNKS
# ---------------------------------------------------------

def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Divide cleaned text into overlapping chunks.

    Default chunk size:
    800 characters

    Default overlap:
    120 characters
    """

    if not isinstance(text, str):
        raise TypeError(
            "The text value must be a string."
        )

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    if not text.strip():
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop when the final part of the text is reached.
        if end >= text_length:
            break

        # Move forward while repeating 120 characters.
        start = end - overlap

    return chunks


# ---------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------

def main() -> None:
    """
    Test PDF finding, extraction, cleaning and chunking.
    """

    # Find all PDFs inside the documents folder.
    pdf_files = find_pdf_files()

    print(
        f"PDF files found: {len(pdf_files)}"
    )

    if not pdf_files:
        print(
            "No PDF files were found inside "
            "the documents folder."
        )
        return

    # Test only the first PDF file.
    test_pdf = pdf_files[0]

    print(
        f"\nTesting PDF: {test_pdf.name}"
    )

    # Extract all pages from the selected PDF.
    pages = extract_pdf_pages(
        test_pdf
    )

    print(
        f"Number of pages: {len(pages)}"
    )

    # Find the first page containing readable text.
    first_text_page = None

    for page_data in pages:
        if page_data["text"]:
            first_text_page = page_data
            break

    if first_text_page is None:
        print(
            "No readable text was found "
            "inside this PDF."
        )
        return

    print(
        f"First readable page: "
        f"{first_text_page['page']}"
    )

    # Clean the extracted page text.
    cleaned_text = clean_text(
        first_text_page["text"]
    )

    print(
        f"Cleaned text length: "
        f"{len(cleaned_text)} characters"
    )

    # Divide the cleaned text into chunks.
    chunks = split_text_into_chunks(
        cleaned_text
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # Print only the first two chunks.
    for number, chunk in enumerate(
        chunks[:2],
        start=1,
    ):
        print(
            f"\nCHUNK {number}:\n"
        )

        print(chunk)


# ---------------------------------------------------------
# RUN THE PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main()