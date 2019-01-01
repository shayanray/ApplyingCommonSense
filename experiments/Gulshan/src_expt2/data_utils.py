from config import *
import pandas as pd
import numpy as np
import pickle
import nltk
from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.data import load
from string import punctuation
import os.path


# DATA LOADING #######################################################

def load_data(dataset):
    """
    Build the dataframe from the given dataset

    :param dataset: path to csv data file (train, val or test data)
    :return: dataframe corresponding to the dataset
    """

    assert dataset == train_set or dataset == val_set or dataset == test_set or dataset == test_set_cloze or dataset == train_set_sampled

    if dataset == train_set:
        names = ['id', 'storytitle', 'sen1', 'sen2', 'sen3', 'sen4', 'sen5']
    elif dataset == test_set:
        names = ['sen1', 'sen2', 'sen3', 'sen4', 'sen5_1', 'sen5_2']
    else:
        names = ['id', 'sen1', 'sen2', 'sen3', 'sen4', 'sen5_1', 'sen5_2', 'ans']

    # return pd.read_csv(dataset, index_col='id', names=names, skiprows=1, sep = ';')
    # For sentiment uncomment the following return
    if dataset == test_set:
        return pd.read_csv(dataset, names=names)

    return pd.read_csv(dataset, names=names, skiprows=1)


def get_answers(dataset):
    """
    Get answers from validation or test dataset

    :param dataset: dataframe containing val / test data
    :return: answers array
    """

    assert dataset == val_set or dataset == test_set_cloze or dataset == train_set_sampled

    df = load_data(dataset)

    answers = df['ans'].values

    return answers


# VOCABULARY #######################################################

def word_cleaning(pos_tag_data, punct, stop_words, lemm):
    """
    Clean words in pos-tag dataset

    :param pos_tag_data: train / val / test dataset with pos-tagged words
    :param punct: True to remove punctuation from sentences
    :param stop_words: True to remove stop words from sentences
    :param lemm: True to lemmatize words in sentences
    :return: processed dataset with words processed according to parameters above
    """

    # create array for processed data
    data_processed = np.empty(pos_tag_data.shape, dtype=list)

    # list of words to be removed
    words_to_remove = []
    if punct:
        words_to_remove += list(punctuation)
    if stop_words:
        download('stopwords', quiet=True)
        words_to_remove += list(stopwords.words('english'))

    # download wordnet for lemmatization if needed
    if lemm:
        download('wordnet', quiet=True)

    for i, sentence in np.ndenumerate(pos_tag_data):
        data_processed[i] = []
        for word_pos in sentence:
            word_pos = (word_pos[0].lower(), word_pos[1])
            # lemmatize words to keep
            if word_pos[0] not in words_to_remove:
                if lemm:
                    data_processed[i].append(tuple(lemmatize(word_pos)))
                else:
                    data_processed[i].append(word_pos)

    return data_processed


def lemmatize(word_pos):
    """
    Lemmatize words with Word Net Lemmatizer

    :param word_pos: tuple containing word and its pos-tag
    :return: tuple with lemmatized word and its pos-tag
    """

    lemmatizer = WordNetLemmatizer()

    # get word and pos-tag
    word, pos = word_pos

    # lemmatize word
    new_word_pos = (lemmatizer.lemmatize(word, get_wordnet_pos(pos)), pos)

    return new_word_pos


def get_wordnet_pos(tag):
    """
    Get wordnet pos tag for lemmatization

    :param tag: pos-tag
    :return: wordnet pos-tag
    """

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def count_words(sentences):
    """
    Count words in sentences

    :param sentences: array containing train / val / test sentences
    :return: counter of words
    """

    # initialize counter
    words = Counter()

    # add sentences to counter
    for s in sentences:
        words.update(s)

    print("Found", len(words), "words in dataset.")

    return words


def generate_vocabulary(data):
    """
    Generate vocabulary and save it as pickle

    :param data: array containing train / val / test data
    :return: vocabulary with words and pos tags
    """

    print("Generating vocabulary...")

    words = count_words(data)

    # create vocabulary
    if vocabulary_size is not None:
        vocabulary = dict((x, y) for x, y in words.most_common(vocabulary_size - (1 if sentence_len is None else 2)))
    else:
        vocabulary = words

    # generate index for each word
    for i, k in enumerate(vocabulary.keys()):
        vocabulary[k] = i

    # if sentence length is fixed add pad
    if sentence_len is not None:
        vocabulary.update({pad: len(vocabulary)})

    # if vocabulary size is fixed add unk
    if vocabulary_size is not None:
        vocabulary.update({unk: len(vocabulary)})

    with open(vocabulary_pkl, 'wb') as output:
        pickle.dump(vocabulary, output, pickle.HIGHEST_PROTOCOL)
        print("Vocabulary saved as pkl")

    return vocabulary


def load_vocabulary():
    """
    Load existing vocabulary

    :return: vocabulary
    """

    print("Loading vocabulary... ")

    try:
        with open(vocabulary_pkl, 'rb') as handle:
            vocabulary = pickle.load(handle)
        print("Vocabulary loaded")
    except FileNotFoundError:
        print("Vocabulary not found.")

    return vocabulary


def load_pos_vocabulary():
    """
    Load vocabulary for pos tagging

    :return: pos tagging vocabulary
    """

    print("Loading pos tag vocabulary... ")

    if os.path.isfile(pos_vocabulary_pkl):
        with open(pos_vocabulary_pkl, 'rb') as handle:
            pos_vocabulary = pickle.load(handle)
    else:
        # load tags from nltk
        tagdict = load('help/tagsets/upenn_tagset.pickle').keys()

        # create dictionary
        pos_vocabulary = dict([(list(tagdict)[i], -(i + 1)) for i in range(len(tagdict))])

        with open(pos_vocabulary_pkl, 'wb') as output:
            pickle.dump(pos_vocabulary, output, pickle.HIGHEST_PROTOCOL)
            print("Pos vocabulary saved as pkl")

    return pos_vocabulary


def check_for_unk(data, vocabulary):
    """
    Check for unk in sentence given the vocabulary

    :param data: array containing raw sentences as list of tuples (word, pos-tag)
    :param vocabulary: vocabulary generated during training
    :return: array of sentences where missing words are replaced by unk
    """

    new_data = np.empty(data.shape, dtype=list)

    # replace words not in vocabulary with unk
    for i, sentence in np.ndenumerate(data):
        new_data[i] = []
        for word in sentence:
            new_data[i].append(word if word[0] in vocabulary.keys() else (unk, word[1]))

    return new_data


def get_words_from_indexes(indexes, vocabulary, pos_vocabulary):
    """
    Get words from indexes in the vocabulary

    :param indexes: list of indexes of words in vocabulary
    :param vocabulary: vocabulary
    :return: words corresponding to given indexes
    """

    # map indexes to words in vocabulary
    vocabulary_reverse = {v: k for k, v in vocabulary.items()}
    pos_vocabulary_reverse = {v: k for k, v in pos_vocabulary.items()}

    # retrieve words corresponding to indexes
    if isinstance(indexes, list):
        if isinstance(indexes[0], tuple):
            words = [(vocabulary_reverse[x[0]], pos_vocabulary_reverse[x[1]]) for x in indexes]
        else:
            words = [vocabulary_reverse[x] for x in indexes]
    else:
        if isinstance(indexes, tuple):
            words = (vocabulary_reverse[indexes[0]], pos_vocabulary_reverse[indexes[1]])
        else:
            words = vocabulary_reverse[indexes]
    return words


def get_index_from_tag(tag):
    '''
    Get vocabulary index for some tag
    :param tag: tage of the form '<tag>'
    :return: tag index
    '''
    with open(full_vocabulary_pkl, 'rb') as f:
        vocabulary = pickle.load(f)
    tag_index = vocabulary[tag]
    return tag_index


def get_indexes_from_words(words, vocabulary, pos_vocabulary):
    """
    Get indexes from words in the vocabulary

    :param words: list of words in vocabulary
    :param vocabulary: vocabulary
    :param pos_vocabulary: pos tag vocabulary
    :return: indexes corresponding to given words
    """

    # retrieve indexes corresponding to words
    if isinstance(words, list):
        if isinstance(words[0], tuple):
            indexes = [(vocabulary[x[0]], pos_vocabulary[x[1]]) for x in words]
        else:
            indexes = [vocabulary[x] for x in words]
    else:
        if isinstance(words, tuple):
            indexes = (vocabulary[words[0]], pos_vocabulary[words[1]])
        else:
            indexes = vocabulary[words]

    return indexes


def pad_endings(beginnings, endings):
    """
    Pad endings according to story_len

    :param beginnings: story beginnings containing the first 4 sentences
    :param endings: story endings containing the last sentence
    :return: padded endings
    """

    print("Padding endings...")

    assert len(beginnings) == len(endings)

    padded_endings = np.empty(endings.shape, dtype=list)

    len_beginnings = [sum(len(s) for s in beginning) for beginning in beginnings]

    for i, ending in np.ndenumerate(endings):

        # trim the ending if the story is too long
        if len_beginnings[i[0]] + len(ending) > story_len:
            padded_endings[i] = ending[:story_len - len_beginnings[i[0]]]

        # pad the ending if the story is too short
        else:
            padded_endings[i] = ending
            while len_beginnings[i[0]] + len(padded_endings[i]) < story_len:
                padded_endings[i].append((pad, '.'))

    return beginnings, padded_endings


# DATA STRUCTURES HANDLING #######################################################

def combine_sentences(sentences):
    """
    Combine multiple sentences in one sentence

    :param sentences: array of sentences
    :return: combines sentence
    """

    # get number of stories
    n_stories, *_ = sentences.shape
    print(sentences.shape)
    # combine sentences
    combined = np.empty(n_stories, dtype=list)
    for i in range(n_stories):
        combined[i] = []
        combined[i].extend([sentences[i]])

    print(combined, "\n\n\n\n\n")
    return combined


def combine_story(beginnings, endings):
    """
    Combine beginnings and endings to get stories

    :param beginnings: array of beginnings with the first 4 sentences
    :param endings: array of endings
    :return: array of stories
    """

    assert len(beginnings) == len(endings)
    print("beginnings's shape: {}".format(beginnings.shape))
    print("endings shape : {}".format(endings.shape))

    # get number of stories
    n_stories, *_ = beginnings.shape

    # combine beginnings sentences
    beginnings = combine_sentences(beginnings)

    # create stories
    stories = np.empty(n_stories, dtype=list)
    for i in range(n_stories):
        stories[i] = []
        stories[i].extend([beginnings[i], endings[i]])

    return stories


def filter_words(dataset):
    """
    Get sentences from pos-tagged dataset

    :param dataset: dataset containing lists of tuples (word, pos-tag)
    :return: array of sentences
    """

    # filter pos-tag dataset by words
    filtered_words = np.empty(dataset.shape, dtype=list)
    for i, sentence in np.ndenumerate(dataset):
        filtered_words[i] = [word[0] for word in sentence]

    return filtered_words


def get_context_sentence(contexts, i):
    """
    Get the i-th sentences from the contexts
    :param contexts: contexts array with 4 sentences per story
    :param i: i-th sentence we want to filter
    :return: filtered context matrix with i-th sentences
    """
    assert i in range(1, 5)

    return contexts[:, i - 1]


def generate_vocab_pos(pos_data):
    """
    Generate a pos_vocabulary file and save it in pkl form
    :param pos_data:
    :return:
    """

    matrix = np.load(pos_data)

    numrows = matrix.shape[0]
    numsen = matrix.shape[1]

    list_pos = list()

    for item in matrix[0:numrows, 1:numsen]:
        for sentence in item[0:numrows]:
            list_pos = list_pos + sentence[:, 1].tolist()
    # print(list(set(list_pos)))

    # creating dictionary with pos_tags, using negative numbers
    pos_dic = dict(enumerate(list(set(list_pos))))
    pos_dictionary = {v: -(k + 1) for k, v in pos_dic.items()}

    print(pos_dictionary)

    with open(pos_vocabulary_pkl, 'wb') as output:
        pickle.dump(pos_dictionary, output, pickle.HIGHEST_PROTOCOL)
        print("Pos vocabulary saved as pkl")

    return pos_dictionary


def generate_vocab_pos_upenn():
    # Getting tags from upenn_tagset
    nltk.download('tagsets', quiet=True)
    tagdict = load('help/tagsets/upenn_tagset.pickle')

    # creating dictionary with pos_tags, using negative numbers
    pos_dic = dict(enumerate(list(set(tagdict.keys()))))
    pos_dictionary = {v: -(k + 1) for k, v in pos_dic.items()}

    # with open(pos_vocabulary_pkl , 'wb') as output:
    #     pickle.dump(pos_dictionary, output, pickle.HIGHEST_PROTOCOL)
    #     print("Pos vocabulary saved as pkl")

    return pos_dictionary


def merge_vocab(vocab1, vocab2):
    """
    Merges two dictionaries in pkl format and saves as new dictionary
    :param vocab1, vocab2: dictionaries to merge in pkl format
    :return: merged dictionary
    """

    with open(vocab1, 'rb') as f:
        data1 = pickle.load(f)

    with open(vocab2, 'rb') as f:
        data2 = pickle.load(f)

    data1.update(data2)

    with open(full_vocabulary_pkl, 'wb') as output:
        pickle.dump(data1, output, pickle.HIGHEST_PROTOCOL)
        print("Full vocabulary saved as pkl")

    return data1
