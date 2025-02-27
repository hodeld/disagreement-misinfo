# disagreement-misinfo
Code to the paper "Disagreement as a way to study misinformation and its effects"


## Converting PDFs to TXT (pdf_to_txt)
This simple app is used to convert letters in PDF format to TXT files using OCR.
The code is based on [ocr2text](https://github.com/writecrow/ocr2text/) which uses [Google Tesseract-OCR](https://github.com/tesseract-ocr/tesseract).


### Installing Tesseract
Tesseract needs to be installed separately. For instructions, see the [ocr2text README](https://github.com/writecrow/ocr2text/blob/main/README.md)

### Running the script
- Place the PDF Files in the `datasets/amct_pdf_files` directory.
- Run the script: `python pdf_to_text/pdf2text.py`
- The converted TXT files will be saved in the `datasets/amct_text_files` directory.
