import pandas as pd
from scipy.stats import spearmanr


import os

from measurement.disagreement import swn_extract_score, neg_sentiment_swn, COL_NR_NEG_PROP, COL_NR_NEG, COL_SWN_COUNT
from measurement.helpers import df_disa_years, plot_editor_letters, plot_disagreement_single_year, getxmlcontent, \
    get_letter_files, OUTPUT_PATH, PQ_LETTERS_PATH, BOOK_LETTERS_PATH, AMCT_TEXT_FILES_PATH, PQ_TEXT_FILES_PATH, merge_pdf_tdm, \
    get_yearly_data_book


def analyze_files_proquest():
    fp_pq = PQ_LETTERS_PATH
    source = PQ_TEXT_FILES_PATH
    col_txt = 'text'
    names_a, dirs_a = get_letter_files(source, '.xml')
    print('files collected')
    df_dirs = pd.Series(dirs_a, index=names_a)
    # df_dirs = df_dirs.sample(1000, ro)
    nr_chunk = 1000
    len_d = len(dirs_a)
    print(len_d)

    df_orig = pd.DataFrame()

    for i in range(0, len_d, nr_chunk):
        print(i)
        i_end = i + nr_chunk

        s_i = df_dirs.iloc[i:i_end]
        dirs, names = s_i.to_list(), s_i.index.to_list()
        new_names, rows = [], []
        for j, (name, fpath) in enumerate(zip(names, dirs)):
            if (j + 1) % 50 == 0:
                print('file nr', i + j + 1)
            if int(name) in df_orig.index:
                continue

            goid, title, date, text, pubtitle = getxmlcontent(fpath)
            if name != goid:
                print('name != goid', name, goid)
                continue
            if text is None:
                print(name, 'text None')
                continue

            df = pd.DataFrame([text], columns=[col_txt])
            try:
                df_senti = swn_extract_score(df, col=col_txt)
            except:
                print('not good', name)
                continue

            df_senti = df_senti.copy()
            assert len(df_senti.index) == 1
            df_senti['text_len'] = len(text)
            df_a = pd.DataFrame([[date, title, pubtitle]], columns=['date', 'title', 'publication_title'])
            df_row = pd.concat([df_a, df_senti], axis='columns')
            df_row.index = [int(name)]
            rows.append(df_row)
        if len(rows) > 0:
            df_all = pd.concat(rows, axis=0)
            df_orig = pd.concat([df_orig, df_all])
            df_orig.to_pickle(fp_pq)
        print('df_orig.shape', df_orig.shape)

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


def analyze():
    # load data
    df = merge_pdf_tdm([COL_NR_NEG, COL_SWN_COUNT])
    dfy = get_yearly_data_book()

    neg_sentiment_swn(df)

    col_neg = COL_NR_NEG_PROP

    # exclude > 2007 because we do not have pdf data for that period
    dfy = dfy[dfy.index <= 2007]
    # merge data
    dfy.index = pd.to_datetime(dfy.index, format='%Y')
    df = df.join(dfy)

    df = df.sort_index()
    df['Total Letters'] = df['Total Letters'].fillna(df['count'])
    col_belief_ct = 'Percent Conspiracies'
    cols = [col_neg, col_belief_ct]
    df = df[cols]

    names = ['Disagreement', 'Engagement in Conspiracy']

    dfi = df_disa_years(df, 1950, 2007)
    print('Corr Book, Neg Sentiment', 1950, 2007,
          spearmanr(dfi[col_neg], dfi[col_belief_ct]))
    plot_editor_letters(dfi, cols=cols, names=names, do_normalize=True)

    dfi = df_disa_years(df, 1950, 2022)
    plot_disagreement_single_year(dfi, col_neg, do_normalize=True)


if __name__ == '__main__':
    #analyze_files_book()
    #analyze_files_proquest()
    analyze()
