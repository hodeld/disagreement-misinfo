import pandas as pd

from measurement.disagreement import swn_extract_score
from measurement.helpers import PQ_LETTERS_PATH, PQ_TEXT_FILES_PATH, get_letter_files, getxmlcontent


def analyze_files_proquest():
    '''Function to analyze the proquest letters.
    This function needs to be executed on the ProQuest server.'''

    print('''Function to analyze the proquest letters.
    This function needs to be executed on the ProQuest server.''')

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
            compression_d = dict(method='zip')
            df_orig.to_csv(fp_pq, compression=compression_d)
        print('df_orig.shape', df_orig.shape)


if __name__ == '__main__':
    analyze_files_proquest()