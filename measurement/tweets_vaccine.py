if __name__ == '__main__':
    date_range_vaxx = ('2020-03', '2023-03', 'month')  # 'week' month
    vaxx_months = ['29_coronavirus_2020_2023']
    #analyze_correlation_years()
    #analyze_plot_sentiment(vaxx_months, date_range=date_range_vaxx)

    swn_to_imdb(redo_harvard=False, redo_analysis=False)
    swn_to_disagreement_UKP()

    #extract_from_extracted('NYT-LtE-2008-2023_keywords_least_neg_letters.csv')  # 'NYT-LtE-2008-2023_keywords_most_neg_letters.csv'