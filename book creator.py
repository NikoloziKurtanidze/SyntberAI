import fitz 

def clean_pdf(input_pdf, output_txt):
    doc = fitz.open(input_pdf)
    with open(output_txt, "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text())
    print("Done! PDF converted to clean text.")

clean_pdf("Full.pdf", "physics_book.txt")
