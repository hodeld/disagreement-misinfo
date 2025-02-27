import os

import pandas as pd

from measurement.disagreement import swn_extract_score
from measurement.helpers import AMCT_TEXT_FILES_PATH, BOOK_LETTERS_PATH, OUTPUT_PATH, get_letter_files


def analyze_files_book():
    def get_txt(fp):
        with open(fp, 'r') as file:
            txt = file.read()
        return txt

    source = AMCT_TEXT_FILES_PATH
    fp_csv = BOOK_LETTERS_PATH
    fp_para = os.path.join(OUTPUT_PATH, f'single_letters.parquet')
    fp_para_temp = os.path.join(OUTPUT_PATH, f'single_letters_temp.parquet')

    col_name = 'name'
    col_txt = 'text'

    df_para = None
    if os.path.isfile(fp_csv):
        df_orig = pd.read_csv(fp_csv, index_col=0)
    else:
        df_orig = pd.DataFrame()

    rows = []
    names, dirs = get_letter_files(source)
    new_names = []
    for name, fpath in zip(names, dirs):
        if int(name) in df_orig.index:
            continue
        new_names.append(int(name))
        text = get_txt(fpath)
        print(name)

        split_str = '\n\n'
        paragraphs = text.split(split_str)
        df = pd.DataFrame(paragraphs, columns=[col_txt])
        df = swn_extract_score(df, col=col_txt)
        df[col_name] = name

        if df_para is not None:
            df_para = pd.concat([df, df_para], axis=0)
        else:
            df_para = df
        df_para.to_parquet(fp_para_temp)

        dfn = df.groupby(col_name).sum(numeric_only=True)
        rows.append(dfn)
    if len(rows) > 0:
        df = pd.concat(rows, axis=0)
        df = pd.concat([df, df_orig], axis=0)

        df_para.to_parquet(fp_para)
        os.remove(fp_para_temp)
        df.to_csv(fp_csv)


if __name__ == '__main__':
    analyze_files_book()