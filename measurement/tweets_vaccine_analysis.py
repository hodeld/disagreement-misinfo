from disagreement import SWN_COLUMNS, swn_extract_score, COL_NR_NEG, neg_sentiment_swn, COL_NR_NEG_PROP
from helpers import get_dataset_parts, copy_orig_files, COL_LEMMA, get_dataframe, \
    transfer_results_to_main_df, save_main_df, TWEETS_AGG_PATH, TWEET_FILES
from misinformation import COL_IS_MISINFO, detect_vaccine_misinfo


def analyze_sentiment_tweets(datasets, date_range, k_sample, calc_col):
    """datatsets: list of Twitter file names with ID as index in pd.DataFrame parquet format.
    The function creates a copy (*_modified*) for each file which includes the sentiment scores for each Tweet.
    date_range: tuple with start and end date as string.
    """
    # load dict with Twitter filename and date ranges
    filename_d = get_dataset_parts(datasets, date_range)

    # make a copy of the original files
    filename_d = copy_orig_files(filename_d)

    date_col = 'created_at'
    tweet_col = 'tweet'
    cols_to_save = SWN_COLUMNS + [COL_LEMMA]

    analyzed_files_d = {}
    for filename, dates_all in filename_d.items():
        df_main = get_dataframe(filename)
        nr_ids = 0
        file_d = {}
        for st_str, et_str in dates_all:
            df = df_main[(df_main[date_col] > st_str) & (df_main[date_col] < et_str)].copy()
            if len(df.index) < k_sample:
                continue
            nr_ids += k_sample
            print(filename, st_str, et_str)
            if calc_col in df.columns and df[~df[calc_col].isna()][calc_col].count() >= k_sample:
                df = df[~df[calc_col].isna()].sample(n=k_sample, random_state=1)
                id_freq = df.index
                file_d[st_str] = id_freq
                print(filename, st_str, et_str, 'already analyzed')
                continue

            df = df.sample(n=k_sample, random_state=1)
            swn_extract_score(df, tweet_col)
            transfer_results_to_main_df(df_main, df, cols_to_save)
            save_main_df(filename, df_main)
            id_freq = df.index
            file_d[st_str] = id_freq
        if len(file_d) > 0:
            analyzed_files_d[filename] = file_d
        df_main = df_main[~df_main[calc_col].isna()]
        assert len(df_main.index) == nr_ids
        save_main_df(filename, df_main)
    return analyzed_files_d


def analyze_misinfo_tweets(analyzed_files_d):
    for filename in analyzed_files_d.keys():
        df = get_dataframe(filename)
        if COL_IS_MISINFO in df.columns and df[COL_IS_MISINFO].notna().all():
            continue
        detect_vaccine_misinfo(df, COL_LEMMA)
        save_main_df(filename, df)


def aggregate_sentiment_tweets(analyzed_files_d, k_sample):
    results = []
    for filename, file_d in analyzed_files_d.items():
        df_main = get_dataframe(filename)
        for st_str, idx in file_d.items():
            df = df_main.loc[idx]
            df = df.sample(n=k_sample, random_state=1)
            neg_sentiment_swn(df)
            neg_mean = df[COL_NR_NEG_PROP].mean()
            nr_misinfo = len(df[df[COL_IS_MISINFO] > .5].index)
            print(st_str, neg_mean)
            results.append({'date': st_str, 'neg_swn': neg_mean, 'nr_tweets': len(df.index), 'nr_misinfo': nr_misinfo})
    df = pd.DataFrame(results)
    df.index = pd.to_datetime(df['date'])
    return df


def tweet_analysis():
    fp = TWEETS_AGG_PATH
    datasets = TWEET_FILES

    col_calc = COL_NR_NEG
    k_sample = 10000
    date_range = ('2020-03', '2023-03', 'month')

    print('Running analysis...')
    analyzed_file_d = analyze_sentiment_tweets(datasets, date_range, k_sample, col_calc)
    analyze_misinfo_tweets(analyzed_file_d)
    df = aggregate_sentiment_tweets(analyzed_file_d, k_sample)
    df.to_csv(fp)


if __name__ == '__main__':
    tweet_analysis()
