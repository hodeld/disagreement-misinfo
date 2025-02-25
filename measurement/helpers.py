import pandas as pd
import re
from datetime import datetime
from lxml import etree
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import matplotlib as plt
import os


DATASETS_PATH = os.path.abspath('../datasets/')
OUTPUT_PATH = os.path.abspath('../data_output/')
PQ_PATH = os.path.join(OUTPUT_PATH, 'proquest_letters.p')
BOOK_PATH = os.path.join(OUTPUT_PATH, 'pdf_letters.p')
AMCT_TEXT_FILES_PATH = os.path.join(DATASETS_PATH, 'amct_text_files')
PQ_TEXT_FILES_PATH = os.path.join(DATASETS_PATH, 'pq_text_files')


COL_POS_TAG = 'pos_tags'
COL_LEMMA = 'after_lemmatization'
REMOVE_SPECIALCHR = True
REMOVE_STOPWORDS = True



def df_disa_years(df, y1_int, y2_int):
    y1, y2 = pd.to_datetime(y1_int, format='%Y'), pd.to_datetime(y2_int, format='%Y')
    dfi = df[(df.index >= y1) & (df.index <= y2)]
    return dfi


def plot_editor_letters(df, cols, names, second_axis=1, do_normalize=False, subtitle=None, title='Engagement in CT & Negative Sentiment'):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))  # x, y

    xlabel = 'Years'
    colors = list([plt.cm.tab20c(i) for i in [0, 4, 12, 16, 1, 5, 13, 17]])

    x_vals = df.index
    for i, (col, n, cl) in enumerate(zip(cols, names, colors)):
        y = df[col] * 100
        label = f'% {n}'
        if second_axis and i != second_axis and do_normalize:
            y = normalize_vals(y)
            label = n
        if second_axis and i == second_axis:
            axi = ax.twinx()  # instantiate a second axes that shares the same x-axis
        else:
            axi = ax
        axi.plot(x_vals, y, label=n, color=cl)
        axi.set_ylabel(label, color=cl)
    ax.set(xlabel=xlabel)
    fig.legend()

    plt.show()


def plot_disagreement_single(df, col, do_normalize=True):
    plt.rcParams.update({'font.size': 14})
    name = 'Disagreement'
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    x_vals = df.index
    y = df[col]
    y = normalize_vals(y)
    ax.plot(x_vals, y, label=name, color='black')
    ax.set_ylabel(f'{name}', fontsize=16)
    return fig, ax


def plot_disagreement_single_year(df, col, do_normalize=True):
    fig, ax = plot_disagreement_single(df, col, do_normalize)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Year', fontsize=16)

    fig.tight_layout(h_pad=7.0,)
    plt.show()


def normalize_vals(s):
    return (s - s.min()) / (s.max() - s.min())


def get_letter_files(source, file_ext='.txt'):
    names, dirs = [], []
    for dirpath, dirnames, files in os.walk(source):
        for i, file in enumerate(files):
            filename, file_extension = os.path.splitext(file)
            if 'dragged' in filename or (file_extension.lower() != file_ext) :
                continue
            names.append(filename)
            source_path = os.path.join(dirpath, file)
            dirs.append(source_path)
    return names, dirs


def merge_pdf_tdm():
    fp_csv = BOOK_PATH
    df = pd.read_csv(fp_csv, index_col=0)

    # add missing rows
    new_index = pd.Series(range(2023 + 1 - 1870)) + 1870
    df = df.reindex(new_index.values)

    df.index.name = 'year'
    df.index = pd.to_datetime(df.index, format='%Y')

    files_tdm = [FNAME_TDM_SWN_19thc, FNAME_TDM_SWN_08_23]
    df_td = concat_tdm_to_df(files_tdm)

    df_td = aggregate_years(df_td, cols_sum=[COL_NR_NEG, COL_SWN_COUNT])
    sfx_tdm, sfx_pdf = '_tdm', '_pdf'
    suffixes = (sfx_pdf, sfx_tdm)
    df = df.merge(df_td, on='year', how='left', suffixes=suffixes)

    for col in cols_vals:
        df[col] = df[col + sfx_pdf].fillna(df[col + sfx_tdm])
        # df[col] = df[col + sfx_tdm]
    return df


def strip_html_tags(text):
    """Function to strip html tags from text portion"""
    stripped = BeautifulSoup(text).get_text().replace('\n', ' ').replace('\\', '').strip()
    return stripped


def getxmlcontent(fpath, strip_html=True):
    try:
        tree = etree.parse(fpath)
        root = tree.getroot()

        if root.find('.//GOID') is not None:
            goid = root.find('.//GOID').text
        else:
            goid = None

        if root.find('.//Title') is not None:
            title = root.find('.//Title').text
        else:
            title = None

        if root.find('.//SortTitle') is not None:
            pubtitle = root.find('.//SortTitle').text
        else:
            pubtitle = None

        if root.find('.//NumericDate') is not None:
            date = root.find('.//NumericDate').text
        else:
            date = None

        if root.find('.//FullText') is not None:
            text = root.find('.//FullText').text

        elif root.find('.//HiddenText') is not None:
            text = root.find('.//HiddenText').text

        elif root.find('.//Text') is not None:
            text = root.find('.//Text').text

        else:
            text = None

        # Strip html from text portion
        if text is not None and strip_html == True:
            text = strip_html_tags(text)

    except Exception as e:
        print(f"Error while parsing file {fpath}: {e}")

    return goid, title, date, text, pubtitle



def preprocessing(df, orig_col_name, redo=False,
                  rm_spechr=REMOVE_SPECIALCHR, rm_stopwords=REMOVE_STOPWORDS,
                  ):
    # from https://www.kaggle.com/code/yommnamohamed/sentiment-analysis-using-sentiwordnet/notebook
    lemmatizer = WordNetLemmatizer()

    def special_characters_data(data, name):
        # Proprocessing the data
        data[name] = data[name].str.lower()
        # Code to remove the Hashtags from the text
        data[name] = data[name].apply(lambda x: re.sub(r'\B#\S+', '', x))
        # Code to remove the links from the text
        data[name] = data[name].apply(lambda x: re.sub(r"http\S+", "", x))
        # Code to remove the Special characters from the text
        data[name] = data[name].apply(lambda x: ' '.join(re.findall(r'\w+', x)))
        # Code to substitute the multiple spaces with single spaces
        data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
        # Code to remove all the single characters in the text
        data[name] = data[name].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))
        # Remove the twitter handlers
        data[name] = data[name].apply(lambda x: re.sub('@[^\s]+', '', x))

    # Function to tokenize and remove the stopwords
    stop_words = set(stopwords.words('english'))
    def rem_stopwords_tokenize(data, name):

        def getting(sen):
            example_sent = sen

            word_tokens = word_tokenize(example_sent)

            filtered_sentence = [w for w in word_tokens if not w in stop_words]

            return filtered_sentence

        # Using "getting(sen)" function to append edited sentence to data
        x = []
        for i in data[name].values:
            x.append(getting(i))
        data[name] = x

    def lemmatization(data, name):
        def getting2(sen_list):
            output_sentence = []
            # word_tokens2 = word_tokenize(example) # already tokenized
            lemmatized_output = [lemmatizer.lemmatize(w) for w in sen_list]

            # Remove characters which have length less than 2
            without_single_chr = [word for word in lemmatized_output if len(word) > 2]
            # Remove numbers
            cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]

            return cleaned_data_title

        # Using "getting2(sen)" function to append edited sentence to data
        x = []
        for i in data[name].values:
            x.append(getting2(i))
        data[name] = x

    def make_sentences(data, name):
        data[name] = data[name].apply(lambda x: ' '.join([i + ' ' for i in x]))
        # Removing double spaces if created
        data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
    col_name_after = COL_LEMMA
    if redo or col_name_after not in df.columns:
        col_name = orig_col_name + '_processed'
        df[col_name] = df[orig_col_name].copy()
        print('start preprocessing and lemmatizing')
        # Using the preprocessing function to preprocess the hotel data
        if rm_spechr:
            special_characters_data(df, col_name)
        # Using tokenizer and removing the stopwords
        if rm_stopwords:
            rem_stopwords_tokenize(df, col_name)
        else:
            df[col_name] = df[col_name].apply(word_tokenize)

        # Edits After Lemmatization
        final_edit = df[col_name].copy()
        df[col_name_after] = final_edit

        # Converting all the texts back to sentences
        make_sentences(df, col_name)

        # Using the Lemmatization function to lemmatize the data
        lemmatization(df, col_name_after)
        # Converting all the texts back to sentences
        make_sentences(df, col_name_after)
        return True
    return False


def tokenize_pos_tagging(df, redo=False):
    col_postags = COL_POS_TAG
    col_tok_lens = 'token_lens'
    for col in (col_postags, col_tok_lens):
        if redo or col not in df.columns:
            df[col] = None # create columns

    # return df where one of columns is None
    df_pos = df[(df[col_postags].isna() | df[col_tok_lens].isna())]
    df_pos = df_pos[[col_postags, col_tok_lens]].copy()
    if len(df_pos.index) > 0:
        postagging, tok_lens = [], []

        print('start tokenization')
        now = datetime.now()
        for k, review in enumerate(df[COL_LEMMA]):
            tok_list = word_tokenize(review)
            postagging.append(nltk.pos_tag(tok_list))
            tok_lens.append(len(tok_list))  # slow part, tags all words according position of speech
            if (k + 1) % 2000 == 0:
                print(f'{k + 1} texts tagged', datetime.now() - now)

        df_pos[col_postags] = postagging
        df_pos[col_tok_lens] = tok_lens
        df.update(df_pos)
    else:
        print('already tokenized and pos-tagged. Using values from DF')
    postagging, tok_lens = df[col_postags], df[col_tok_lens]

    return postagging, tok_lens



