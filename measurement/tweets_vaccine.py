import os
import pandas as pd
from scipy.stats import spearmanr

from helpers import TWEETS_AGG_PATH, plot_disagreement_misinfo, \
    plot_disagreement_single_month
from tweets_vaccine_analysis import tweet_analysis


def compare_tweets():
    fp = TWEETS_AGG_PATH

    cols_plot = ['neg_swn', 'nr_misinfo']

    if os.path.exists(fp) is False:
        print('file not found', fp)
        print('Running analysis...')
        tweet_analysis()

    # load data
    df = pd.read_csv(fp)
    df.index = pd.to_datetime(df['date'])

    corr = spearmanr(df[cols_plot[0]], df[cols_plot[1]])
    print('Spearmans corr', corr)
    names = ['Disagreement', 'Misinformation']
    plot_disagreement_misinfo(df, cols_plot, names, do_normalize=True)
    plot_disagreement_single_month(df, cols_plot[0], do_normalize=True)


if __name__ == '__main__':
    compare_tweets()
