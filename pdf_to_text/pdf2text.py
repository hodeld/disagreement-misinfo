from os import path
from ocr2text_logic import convert_recursive


DATASET_PATH = path.join(path.dirname(path.abspath(__file__)), 'datasets')
INPUT_P = path.join(DATASET_PATH, 'amct_pdf_files')
OUT_P = path.join(DATASET_PATH, 'amct_text_files')


def main():
    print(INPUT_P, OUT_P)
    count = 0
    count = convert_recursive(INPUT_P, OUT_P, count)
    print(str(count) + ' file(s) converted')


if __name__ == '__main__':
    main()