COL_NR_NEG_PROP = 'prop_nr_neg'
COL_SWN_COUNT = 'swn_c'
COL_NR_NEG = 'nr_neg'
COL_NR_POS = 'nr_pos'

SWN_COLUMNS = [COL_NR_POS, COL_NR_NEG, COL_SWN_COUNT]


def sentiwordnet_extract(df, orig_col_name):
    # from https://www.kaggle.com/code/yommnamohamed/sentiment-analysis-using-sentiwordnet/notebook

    preprocessing(df, orig_col_name)
    postagging, tok_lens = tokenize_pos_tagging(df)
    def penn_to_wn(tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None

    # Returns list of pos-neg and objective score. But returns empty list if not present in senti wordnet.
    def get_sentiment(word, tag):
        wn_tag = penn_to_wn(tag)

        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            return []

        # Lemmatization already lemmatized in preprocessing
        # Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet.
        # Synset instances are the groupings of synonymous words that express the same concept.
        # Some of the words have only one Synset and some have several.
        synsets = wn.synsets(word, pos=wn_tag)  # 'fact.n.02 02: means 2nd most meaning of word fact
        if not synsets:
            return []

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        return [synset.name(), swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score(), word]
    def reset_scores():
        nonlocal pos, neg, count
        pos = neg = count = 0
        return pos, neg, count

    senti_score = []
    pos, neg,  count = reset_scores()
    print('start senti scores')
    negatives, positives = [], []
    columns = SWN_COLUMNS

    for k, (pos_val, tok_len) in enumerate(zip(postagging, tok_lens)):
        senti_val = [get_sentiment(x, y) for (x, y) in pos_val]
        for score in senti_val:
            if len(score) > 1:
                posi = score[1]
                negi = score[2]
                negatives.append(negi)
                positives.append(posi)
                pos += posi  # positive score is stored at 2nd position
                neg += negi   # negative score is stored at 3rd position
                count += 1  # count words found
        if k % 10000 == 0:
            print(f'{k} scored')
        senti_score.append((pos, neg, count, tok_len))
        reset_scores()
    print('mean positives', sum(positives) / len(positives))
    df_scores = pd.DataFrame(senti_score, columns=columns, index=df.index)
    return df_scores


def swn_extract_score(df, col):
    df_scores = sentiwordnet_extract(df, col)
    columns = df_scores.columns.to_list()
    df.loc[:, columns] = df_scores[columns]
    return df


def neg_sentiment_swn(df):
    ":returns proportion of negative texts"
    df[COL_NR_NEG_PROP] = df[COL_NR_NEG] / df[COL_SWN_COUNT]
