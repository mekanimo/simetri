"""Extract plain text from a PDF into a text file."""

import pymupdf as fitz


def extract_pdf_text(pdf_path, text_path):
    """Extract the text from the given pdf file and write it to a text file
    in the given text_path"""
    doc = fitz.open(pdf_path)
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())
        text = "\n".join(parts)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
    finally:
        doc.close()
        print("Done!")


pdf_path = r"C:\FB\farbige parkette\farbige parkette no images_bw_ocred.pdf"
text_path = r"C:\FB\farbige parkette\farbige_parkette_text.txt"

extract_pdf_text(pdf_path, text_path)
