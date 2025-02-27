import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline

from measurement.helpers import VACCINE_MISINFO_MODEL

COL_IS_MISINFO = 'is_misinfo'
COL_SENTI_BERT_L, COL_SENTI_BERT_S = 'bert_cf_label', 'bert_cf_score'
COL_NEG_BERT_S = COL_SENTI_BERT_S + '_neg'
COL_SENTI_BERT_L_LEMMA, COL_SENTI_BERT_S_LEMMA = COL_SENTI_BERT_L + 'lemma', COL_SENTI_BERT_S + 'lemma'
COL_MISINFO_SCORE = 'misinfo_score'


def detect_misinfo(df, col, model, tokenizer, predict):
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)
    tokenizer_kwargs = {'truncation': True, 'max_length': 512}
    nr_items = 1000
    is_misinfo_c = COL_IS_MISINFO
    df[is_misinfo_c] = 0
    icol_misinfo = df.columns.get_loc(is_misinfo_c)
    icol_txt = df.columns.get_loc(col)
    for i in range(0, len(df.index), nr_items):
        print(i)
        tweets = df.iloc[i:i + nr_items, icol_txt]
        ser = pd.Series(pipe(tweets.tolist(), **tokenizer_kwargs))
        ser_mi = ser.apply(predict)
        df.iloc[i:i + nr_items, icol_misinfo] = ser_mi.tolist()
    nrmisinfo = len(df[df[is_misinfo_c] > .5].index)
    return nrmisinfo


def detect_vaccine_misinfo(df, col):
    label_name = 'is_misinfo'

    def predict(dicti):
        score = dicti[0]['score']
        is_misinfo = score if dicti[0]['label'] == label_name else 1 - score
        return is_misinfo

    tokenizer = AutoTokenizer.from_pretrained('hodeld/Anti-Vax-Misinfo-Detector')
    model = AutoModelForSequenceClassification.from_pretrained('hodeld/Anti-Vax-Misinfo-Detector')
    return detect_misinfo(df, col, model, tokenizer, predict)


def calc_misinfo(df):
    def set_val(row):
        return 1 if row > .5 else 0

    df[COL_MISINFO_SCORE] = df[COL_IS_MISINFO].apply(set_val)
