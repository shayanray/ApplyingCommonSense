from data_utils import *
from sentiment import *

def pos_tagging_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)


def pos_tag_dataset(dataset, separate=False):
    """
    Saves two files containing pos-tagged sentences:
    1) array of size #stories/rows by 5 (story id, sen1 to sen4)
    2) array of size #stories/rows by 2/3 (story id, sen5, sen6 if exists).
    Each column (not id) is a tockenized and pos-tagged sentence of the form (ie: [[Kelly, RB], [studied, VBN], [.,.]])

    :param dataset: dataframe containing train / val / test data
    :param separate: save beginning sentences as one entry per array row or as 4 seperate sentences/entries in array
    :return: pos tagged beginning and ending matrices
    """

    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

    # load data from csv
    data_original = load_data(dataset)

    # # TODO Remove
    # data_original = data_original.head(10)

    # Removing story title if exists, to have the first sentence as the first column
    #data_original.drop(columns=[c for c in data_original.columns if 'title' in c], inplace=True)

    # Counter to know how many sentences have been processed
    story_number = 0
    total_stories = len(data_original)

    # Dealing with sentence endings: if not training set, then there are two ending sentences
    if dataset == train_set:
        pos_end = pd.DataFrame(columns=['id', 'sen5'])
    else:
        pos_end = pd.DataFrame(columns=['id', 'sen5', 'sen6'])

    # Dealing with story beginning: either group pos tags as one array or seperate per sentence
    print("Doing separate pos-tagging")
    if separate:
        pos_begin = pd.DataFrame(columns=['id', 'sen1', 'sen2', 'sen3', 'sen4'])
        for index, row in data_original.iterrows():
            if dataset == train_set_sampled:
                pos_begin.loc[index] = [index,
                                        np.asarray(pos_tagging_text(row.iloc[1]), object),
                                        np.asarray(pos_tagging_text(row.iloc[2]), object),
                                        np.asarray(pos_tagging_text(row.iloc[3]), object),
                                        np.asarray(pos_tagging_text(row.iloc[4]), object)
                                        ]
            else:
                pos_begin.loc[index] = [index,
                                        np.asarray(pos_tagging_text(row.iloc[0]), object),
                                        np.asarray(pos_tagging_text(row.iloc[1]), object),
                                        np.asarray(pos_tagging_text(row.iloc[2]), object),
                                        np.asarray(pos_tagging_text(row.iloc[3]), object)
                                        ]

            if dataset == train_set:
                pos_end.loc[index] = [index, np.asarray(pos_tagging_text(row.iloc[4]))]
            elif dataset == train_set_sampled:
                pos_end.loc[index] = [index, np.asarray(pos_tagging_text(row.iloc[5])),
                                      np.asarray(pos_tagging_text(row.iloc[6]))]
            else:
                pos_end.loc[index] = [index, np.asarray(pos_tagging_text(row.iloc[4])),
                                          np.asarray(pos_tagging_text(row.iloc[5]))]

            story_number = story_number + 1

            if story_number % 1000 == 0:
                print("Processed ", story_number, "/", total_stories)

        pos_begin = np.asarray(pos_begin)
        pos_end = np.asarray(pos_end)

        print("Saving pos tagged corpus..")
        # Saving models in two data files
        cur_dir = os.path.splitext(dataset)[0]
        path_begin = cur_dir + "_pos_begin"
        path_end = cur_dir + "_pos_end"


    else:
        pos_begin = pd.DataFrame(columns=['id', 'sen'])
        for index, row in data_original.iterrows():
            pos_begin.loc[index] = [index,
                                    np.concatenate((np.asarray(pos_tagging_text(row.iloc[0]), object),
                                                    np.asarray(pos_tagging_text(row.iloc[1]), object),
                                                    np.asarray(pos_tagging_text(row.iloc[2]), object),
                                                    np.asarray(pos_tagging_text(row.iloc[3]), object)
                                                    ))]
            if dataset == train_set:
                pos_end.loc[index] = [index, np.asarray(pos_tagging_text(row.iloc[4]))]
            else:
                pos_end.loc[index] = [index, np.asarray(pos_tagging_text(row.iloc[4])),
                                      np.asarray(pos_tagging_text(row.iloc[5]))]

            story_number = story_number + 1

            if story_number % 1000 == 0:
                print("Processed ", story_number, "/", total_stories)

        pos_begin = np.asarray(pos_begin)
        pos_end = np.asarray(pos_end)

        print("Saving pos tagged corpus..")
        # Saving models in two data files
        cur_dir = os.path.splitext(dataset)[0]  # TODO Remove if unused
        # cur_dir = data_folder + dataset
        path_begin = cur_dir + "_pos_begin_together"
        path_end = cur_dir + "_pos_end_together"

    # Saving pos tagged corpus
    np.save(path_begin, pos_begin)
    np.save(path_end, pos_end)
    print("Saved the pos tagged corpus successfully as {} and {}".format(path_begin, path_end))

    # To load dataset, then do np.load(train_pos_begin)

    return pos_begin, pos_end


def preprocess(pos_begin, pos_end, test=False, pad='ending', punct=True, stop_words=True, lemm=True):
    """
    Preprocess pos-tagged data

    :param pos_begin: pos-tagged context sentences in train / val / test data
    :param pos_end: pos-tagged ending sentences in train / val / test data
    :param pad: 'ending' to pad the endings or None
    :param punct: True to remove punctuation from sentences
    :param stop_words: True to remove stop words from sentences
    :param lemm: True to lemmatize words in sentences
    :param test: False if train data, True if val / test data
    :return: processed pos-tagged beginning and ending data
    """

    # remove id from datasets
    pos_begin = np.delete(pos_begin, 0, axis=1)
    pos_end = np.delete(pos_end, 0, axis=1)

    # get number of stories
    n_stories = len(pos_begin)

    # clean words in sentences
    begin_processed = word_cleaning(pos_begin, punct, stop_words, lemm)
    end_processed = word_cleaning(pos_end, punct, stop_words, lemm)

    # generate vocabulary if training
    if not test:
        n_stories, n_beginnings = begin_processed.shape
        filtered_begin = np.reshape(filter_words(begin_processed), (n_stories * n_beginnings))
        filtered_end = filter_words(end_processed)
        filtered = np.append(filtered_begin, filtered_end)
        vocabulary = generate_vocabulary(filtered)
    # load vocabulary if testing
    else:
        vocabulary = load_vocabulary()

    pos_vocabulary = load_pos_vocabulary()

    if not test:
        merge_vocab(vocabulary_pkl, pos_vocabulary_pkl)

    # replace words not in vocab with unk
    begin_processed = check_for_unk(begin_processed, vocabulary)
    end_processed = check_for_unk(end_processed, vocabulary)

    # pad ending if needed
    if pad == 'ending':
        begin_processed, end_processed = pad_endings(begin_processed, end_processed)

    # map words to vocabulary indexes
    for i in range(n_stories):
        begin_processed[i] = [[get_indexes_from_words(sen, vocabulary, pos_vocabulary) for sen in story] for story in
                              begin_processed[i]]
        end_processed[i] = [[get_indexes_from_words(sen, vocabulary, pos_vocabulary) for sen in story] for story in
                            end_processed[i]]

    return begin_processed, end_processed

# TODO REmove if not used
def combine_matrix_cols(array):
    return combined_matrix


# TODO Remove if not used
def open_csv_asmatrix(datafile):
    print("Loading ", datafile)
    file_csv = pd.read_csv(datafile)
    file = np.asarray(file_csv)
    print("Loaded ", datafile, " successfully!")
    return file


def load_train_val_datasets_pos_tagged(together = True, stop_words=False, lemm=True):

    if together:
        print("Loading train set together..")
        pos_train_begin, pos_train_end = preprocess(pos_begin = np.load(train_pos_context_tog), pos_end = np.load(train_pos_end_tog), test=False, pad='ending', punct=True,
                                                            stop_words=stop_words, lemm=lemm)
        print("Loading validation set together..")
        pos_val_begin, pos_val_end = preprocess(pos_begin = np.load(val_pos_context_tog), pos_end = np.load(val_pos_end_tog), test=True, pad='ending', punct=True,
                                                        stop_words=stop_words, lemm=lemm)
    else:
        print("Loading train set separate..")
        pos_train_begin, pos_train_end = preprocess(pos_begin = np.load(train_pos_begin), pos_end = np.load(train_pos_end), test=False, pad='ending', punct=True,
                                                            stop_words=stop_words, lemm=lemm)
        print("Loading validation set separate..")
        pos_val_begin, pos_val_end = preprocess(pos_begin = np.load(val_pos_begin), pos_end = np.load(val_pos_end), test=True, pad='ending', punct=True,
                                                        stop_words=stop_words, lemm=lemm)

    return pos_train_begin, pos_train_end, pos_val_begin, pos_val_end

def generate_binary_verifiers(dataset = None):

    binary_verifiers = []

    ver_val_set = get_answers(dataset)
    # print("Verifier")
    # print(len(ver_val_set))
    # print(dataset)
    for value in ver_val_set:
        if value == 1:
            binary_verifiers.append([1, 0]) #Correct ending is the first one
        else:
            binary_verifiers.append([0, 1]) #Incorrect ending is the second one
    #print(binary_verifiers)
    return binary_verifiers

def eliminate_id(dataset):   
    dataset_no_id = []
    sentence_in_stories = len(dataset[0])
    total_stories = len(dataset)
    for story_number in range(total_stories):
        new_story = dataset[story_number][1:sentence_in_stories] # delete ids stories
        story_number = story_number + 1
        dataset_no_id.append(new_story)
    return dataset_no_id


def sentences_to_sentiments(contexts):
    all_scores = []
    context_n = 0
    for context in contexts:
        scores = []
        #print("LEN ", len(context))
        for sentence in context:
            #print("\nSENTENCE\n", sentence)
            #print(" ".join(word for word in sentence))
            #sentence_sentiment(" ".join(word for word in sentence))
            scores.extend(np.around(sentence_sentiment(" ".join(word for word in sentence))*1000+1000))
        #print("FINAL UNIQUE ARRAY ", scores)
        context_n = context_n +1
        if context_n%1000 == 0:
            print("Contexts to sentiments ",context_n,"/",len(contexts))
        all_scores.append(scores)
    return all_scores

def endings_to_sentiments(endings):
    all_scores = []
    ending_n = 0
    for batch_endings in endings:
        scores_ending = []
        for sentence in batch_endings:
            #print("\nHERE ENDING\n\n", sentence)
            scores_ending.append(np.around(sentence_sentiment(" ".join(word for word in sentence))*1000+1000))
        all_scores.append(scores_ending)
        #print("FINAL TWO UNIQUE ARRAY ", scores_ending)
        ending_n = ending_n +1
        if ending_n%1000 == 0:
            print("Endings to sentiments ",ending_n,"/",len(endings))
    return np.asarray(all_scores)

def train_verifier(train_dataset):
    verifiers = len(train_dataset)
    ver_array = []
    for i in range(verifiers):
        ver_array.append([1,0])
    return np.asarray(ver_array)



def full_sentence_story(dataset, ending_separate=False):
    '''
    creates an array that merges columns containing sentences into one column
    :return: 1-d array containing entire story - shape(#stories, 1)
    '''
    data_original = load_data(dataset)
    fullstory_df = pd.DataFrame(index=None, columns=["full_story"])
    endingstory_df = pd.DataFrame(index=None, columns=["ending_story"])
    if ending_separate==False:
        if dataset==train_set:
            fullstory_df["full_story"] = data_original["sen1"].map(str) + ' ' + data_original["sen2"].map(str) + ' ' + \
                                         data_original["sen3"].map(str) + ' ' + data_original["sen4"].map(str) + ' ' + \
                                         data_original["sen5"].map(str)
        return fullstory_df.values.ravel()
    else:
        if dataset==train_set:
            fullstory_df["full_story"] = data_original["sen1"].map(str) + ' ' + data_original["sen2"].map(str) + ' ' + \
                                         data_original["sen3"].map(str) + ' ' + data_original["sen4"].map(str)
            endingstory_df["ending_story"] = data_original["sen5"].map(str)
        return fullstory_df.values.ravel(), endingstory_df.ravel()

# just trying if works
if __name__ == '__main__':
    pos_tag_dataset(train_set, separate=False)
    pos_tag_dataset(val_set, separate=False)
    pos_tag_dataset(test_set_cloze, separate=False)
    pos_tag_dataset(test_set, separate=False)
    pos_tag_dataset(train_set, separate=True)
    pos_tag_dataset(val_set, separate=True)
    pos_tag_dataset(test_set_cloze, separate=True)
    pos_tag_dataset(test_set, separate=True)
    pos_tag_dataset(train_set_sampled, separate=True)
