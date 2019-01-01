from config import *
from preprocessing import load_data
import nltk
import pandas as pd
import numpy as np
import os
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentence_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sentence)
    scores_array = np.asarray(list(scores.values()))
    #print("SCORES ARE ", np.asarray(list(scores.values())))
    return scores_array


def sentiment_analysis(dataset):

    if dataset == train_set and os.path.isfile(sentiment_train_pkl):
        return load_sentiment(sentiment_train_pkl)

    if dataset == val_set and os.path.isfile(sentiment_val_pkl):
        return load_sentiment(sentiment_val_pkl)

    nltk.download('vader_lexicon', quiet=True)

    # load data from csv
    data_original = load_data(dataset)
    # print(data_original)
    #Only go through the first 10 entries of dataset - Remove for entire dataset
    # data_original = data_original.head(20)

    sid = SentimentIntensityAnalyzer()

    sentiment_score = pd.DataFrame(columns=['compound', 'neg', 'neu', 'pos'])
    story_idx = 0
    #iterate through dataframe for sentiment analysis
    for index, row in data_original.iterrows():
        #print(row)
        story_to_complete = " ".join([row['sen1'], row['sen2'], row['sen3'], row['sen4']])
        #story_to_complete = "'''{0}'''".format(story_to_complete)
        # print(story_to_complete)
        scores = sid.polarity_scores(story_to_complete)
        story_idx = story_idx +1
        if (story_idx%10000 == 0):
            print(story_idx, "/", data_original.shape[0])
        for key in sorted(scores):
            # print('{0}:{1}, '.format(key, scores[key]), end='')
            #print(scores[key])
            sentiment_score.loc[index] = scores

    if dataset == train_set:
        with open(sentiment_train_pkl, 'wb') as output:
            pickle.dump(sentiment_score, output, pickle.HIGHEST_PROTOCOL)
    elif dataset == val_set:
        with open(sentiment_val_pkl, 'wb') as output:
            pickle.dump(sentiment_score, output, pickle.HIGHEST_PROTOCOL)

    return sentiment_score


def load_sentiment(pkl):
    with open(pkl, 'rb') as handle:
        sentiment = pickle.load(handle)
    return sentiment


