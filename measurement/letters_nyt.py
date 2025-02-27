import pandas as pd
from scipy.stats import spearmanr

import os

from disagreement import neg_sentiment_swn, COL_NR_NEG_PROP, COL_NR_NEG, COL_SWN_COUNT
from helpers import df_disa_years, plot_editor_letters, plot_disagreement_single_year, PQ_LETTERS_PATH, BOOK_LETTERS_PATH, \
    merge_pdf_tdm, \
    get_yearly_data_book
from measurement.letters_book import analyze_files_book
from measurement.letters_proquest import analyze_files_proquest


def analyze_letters():
    # load data
    df = merge_pdf_tdm([COL_NR_NEG, COL_SWN_COUNT])
    dfy = get_yearly_data_book()

    col_neg = COL_NR_NEG_PROP

    neg_sentiment_swn(df)

    #  exclude > 2007 because we do not have pdf data for that period
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
    if os.path.exists(BOOK_LETTERS_PATH) is False:
        analyze_files_book()
    if os.path.exists(PQ_LETTERS_PATH) is False:
        analyze_files_proquest()
    analyze_letters()
