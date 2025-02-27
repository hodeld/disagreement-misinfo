import shutil

import pandas as pd
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from lxml import etree
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from matplotlib import pyplot as plt, dates
import os

# Define paths
DATASETS_PATH = os.path.abspath('../datasets/')
OUTPUT_PATH = os.path.abspath('../data_output/')
PQ_LETTERS_PATH = os.path.join(OUTPUT_PATH, 'proquest_letters.csv.zip')
BOOK_LETTERS_PATH = os.path.join(OUTPUT_PATH, 'book_letters.csv')
AMCT_TEXT_FILES_PATH = os.path.join(DATASETS_PATH, 'amct_text_files')
PQ_TEXT_FILES_PATH = os.path.join(DATASETS_PATH, 'pq_text_files')
BOOK_DATA_PATH = os.path.join(DATASETS_PATH, 'yearly_data_from_book.csv')

VACCINE_MISINFO_MODEL = os.path.join(os.path.abspath('../misinfo_model/'), 'vaxx_bert_model')

TWEETS_AGG_PATH = os.path.join(OUTPUT_PATH, 'tweet_disa_misinfo.csv')
TWEETS_DATASETS_PATH = os.path.join(DATASETS_PATH, 'tweet_datasets')

# Define tweet file names as list
TWEET_FILES = ['filenames']

COL_POS_TAG = 'pos_tags'
COL_LEMMA = 'after_lemmatization'
REMOVE_SPECIALCHR = True
REMOVE_STOPWORDS = True


def df_disa_years(df, y1_int, y2_int):
    y1, y2 = pd.to_datetime(y1_int, format='%Y'), pd.to_datetime(y2_int, format='%Y')
    dfi = df[(df.index >= y1) & (df.index <= y2)]
    return dfi


def plot_editor_letters(df, cols, names, second_axis=1, do_normalize=False):
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

    fig.tight_layout(h_pad=7.0, )
    plt.show()


def plot_disagreement_misinfo(df, cols, names, second_axis=1, do_normalize=False):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))  # x, y
    x_vals = df.index
    colors = list([plt.cm.tab20c(i) for i in [0, 4, 12, 16, 1, 5, 13, 17]])
    for i, (c, n, cl) in enumerate(zip(cols, names, colors)):
        y = df[c]
        if second_axis and i != second_axis and do_normalize:
            y = normalize_vals(y)
        if second_axis and i == second_axis:
            axi = ax.twinx()  # instantiate a second axes that shares the same x-axis
        else:
            axi = ax
        axi.plot(x_vals, y, label=n, color=cl)

        if second_axis == 2 and i < 2:
            ylabel = 'Disagreement & Misinformation'
            ax.set_ylabel(ylabel)
        else:
            axi.set_ylabel(f'# {n} calculated', color=cl)
        axi.get_legend_handles_labels()
    fig.legend(loc="upper right", bbox_to_anchor=(.9, .9), )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Year-Month')
    ax.xaxis.set_major_locator(dates.MonthLocator(interval=3))

    topic = 'Vaccines'
    title = f'Disagreement & Misinformation {topic}'
    # fig.suptitle(title)  #
    fig.tight_layout(h_pad=7.0, )  # not together with set position
    plt.show()


def plot_disagreement_single_month(df, col, do_normalize=True):
    fig, ax = plot_disagreement_single(df, col, do_normalize)

    ax.get_legend_handles_labels()

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Year-Month', fontsize=16)
    ax.xaxis.set_major_locator(dates.MonthLocator(interval=3))

    fig.tight_layout(h_pad=7.0, )  # do not set together with set position
    plt.show()


def normalize_vals(s):
    return (s - s.min()) / (s.max() - s.min())


def get_letter_files(source, file_ext='.txt'):
    names, dirs = [], []
    for dirpath, dirnames, files in os.walk(source):
        for i, file in enumerate(files):
            filename, file_extension = os.path.splitext(file)
            if 'dragged' in filename or (file_extension.lower() != file_ext):
                continue
            names.append(filename)
            source_path = os.path.join(dirpath, file)
            dirs.append(source_path)
    return names, dirs


def merge_pdf_tdm(cols_vals):
    fp_csv = BOOK_LETTERS_PATH
    df = pd.read_csv(fp_csv, index_col=0)

    # add missing rows
    new_index = pd.Series(range(2023 + 1 - 1870)) + 1870
    df = df.reindex(new_index.values)
    df.index.name = 'year'
    df.index = pd.to_datetime(df.index, format='%Y')

    df_td = pd.read_csv(PQ_LETTERS_PATH, header=0, index_col=0)
    df_td = aggregate_years(df_td, cols_vals)

    sfx_tdm, sfx_pdf = '_tdm', '_pdf'
    suffixes = (sfx_pdf, sfx_tdm)
    df = df.merge(df_td, on='year', how='left', suffixes=suffixes)

    for col in cols_vals:
        df[col] = df[col + sfx_pdf].fillna(df[col + sfx_tdm])
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


def preprocessing(df, orig_col_name, redo=True,
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
            df[col] = None  # create columns

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


def get_yearly_data_book():
    fp_csv = BOOK_DATA_PATH
    dfy = pd.read_csv(fp_csv, index_col=0)
    dfy.index = dfy.index.astype(int)
    return dfy


def aggregate_years(df, col_vals, time_period='year', col_count='count'):
    agg_dfs = []
    col_year, col_m = year_month_from_string(df, col_date='date')
    if time_period == 'year':
        col, fm = col_year, '%Y'
    else:
        col, fm = col_m, '%Y-%m'
    cols = [col] + col_vals
    df = df[cols]
    df_count = df.copy()
    df_count[col_count] = df_count[col_year]
    df_count = df_count.groupby(col).count()
    df_count = df_count[[col_count]]
    agg_dfs.append(df_count)

    if time_period == 'year':
        df_sum = df.groupby(col).sum(numeric_only=True)
        df_sum = df_sum[col_vals]
        agg_dfs.append(df_sum)
    else:
        df_mean = df.groupby(col).mean(numeric_only=True)
        df_mean = df_mean[col_vals]
        subtitle = 'mean'
        df_mean.columns = [c + '_' + subtitle for c in df_mean.columns]
        agg_dfs.append(df_mean)

    df_agg = pd.concat(agg_dfs, axis=1)
    df_agg.index = pd.to_datetime(df_agg.index, format=fm)
    return df_agg


def year_month_from_string(df, col_date='date'):
    def calc_year(row):
        y = int(row[col_date][:4])
        return y

    def calc_month(row):
        y = row[col_date][:7]
        return y

    col_year = 'year'
    df[col_year] = df.apply(lambda x: calc_year(x), axis=1)
    col_m = 'year-month'
    df[col_m] = df.apply(lambda x: calc_month(x), axis=1)
    return col_year, col_m


def get_dataset_parts(datasets, date_range):
    """Helper function to identify data range for each Twitter dataset."""

    dtf = '%Y-%m-%d %H:%M:%S'
    date_col = 'created_at'
    filename_d = {}
    start_str, end_str, frequency = date_range
    start_date = datetime.strptime(start_str + '-01 00:00:00', dtf) - relativedelta(months=1)
    start_str = start_date.strftime(dtf)
    end_date = datetime.strptime(end_str + '-01', '%Y-%m-%d')
    for filename in datasets:
        df = get_dataframe(filename)
        df_start, df_end = df[date_col].min().replace(tzinfo=None), df[date_col].max().replace(tzinfo=None)
        if df_start > df_end or end_date < df_start:
            continue
        st_str = start_str  #
        nr_m = 0
        while True:
            nr_m += 1
            st_str, et_str, do_continue = get_dates(st_str, end_date, frequency, dtf)
            if do_continue is False:
                break
            dates_all = filename_d.get(filename, [])
            dates_all.append((st_str, et_str))
            filename_d[filename] = dates_all
    return filename_d


def get_dates(s_str, end_dt, freq, dtf):
    dt1 = datetime.strptime(s_str, dtf)
    if dt1 > end_dt:
        return None, None, False
    if freq == 'month':
        rel_delta = relativedelta(months=1)
    elif freq == 'week':
        rel_delta = relativedelta(days=7)
    dt1 += rel_delta
    dt2 = dt1 + rel_delta
    s_str = dt1.strftime(dtf)
    e_str = dt2.strftime(dtf)
    do_cont = True
    return s_str, e_str, do_cont


def get_dataframe(fn):
    fp = get_fp(fn)
    df = pd.read_parquet(fp)
    return df


def get_fp(file_name, file_type='parquet'):
    return os.path.join(TWEETS_DATASETS_PATH, f'{file_name}.{file_type}')


def copy_orig_files(filename_d):
    """makes a copy of the original files"""
    suffix = '_modified'
    new_d = {}
    for key, val in filename_d.items():
        new_key = key + suffix
        new_d[new_key] = val
        dest = get_fp(new_key)
        if os.path.exists(dest):
            continue
        print('copying', key)
        source = get_fp(key)
        shutil.copy(source, dest)
    return new_d


def transfer_results_to_main_df(dfmain, df, cols):
    for col in cols:
        dfmain.loc[df.index, col] = df[col]


def save_main_df(file_name, df):
    fp = get_fp(file_name)
    df.to_parquet(fp)
    print('df saved', file_name)
