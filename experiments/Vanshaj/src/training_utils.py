from copy import deepcopy
from preprocessing import *
from config import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



"""THIS PART UNTIL THE BATCH ITERS SHOULD E MOVED SOMEWHERE ELSE"""

def aggregate_contexts(contexts):
    contexts_aggregated = []
    for context in contexts:
        context_aggregated = []

        for sentence in context:
            context_aggregated = context_aggregated + sentence

        contexts_aggregated.append(context_aggregated)

    return contexts_aggregated


def full_stories_together(contexts, endings, contexts_aggregated = True, validation = False, list_array = False):

    if not contexts_aggregated:
        contexts = aggregate_contexts(contexts)

    full_stories_batches = []

    idx_batch_endings = 0

    for context in contexts:
        story_endings = endings[idx_batch_endings]

        full_story_batch = []
        for ending in story_endings:

            if list_array:
                original_context = deepcopy(context[0])
            else:
                original_context = deepcopy(context)

            full_story = list(original_context) + list(ending)
            lenght = len(full_story)
            if lenght > story_len:
                full_story = full_story[0:story_len]
            else:
                full_story = full_story

            full_story_batch.append(full_story)

        full_stories_batches.append(full_story_batch)
        idx_batch_endings = idx_batch_endings + 1
        if idx_batch_endings%20000 == 0:
            print("Stories combined together ",idx_batch_endings,"/",len(contexts))

    return full_stories_batches


#For this function the datast needs to be pos tagged
def batches_pos_neg_endings(neg_end_obj, endings, batch_size):
    """INPUT:
             neg_end_obj : Needs the negative endings objects created beforehand
             endings : dataset
             batch_size : batch_size - 1 negative endings will be created
        """
    total_stories = len(endings)
    aug_data = []
    ver_aug_data = []
    for story_idx in range(0, total_stories):

        batch_aug_stories, ver_aug_stories = neg_end_obj.words_substitution_approach(ending_story = endings[story_idx], batch_size = batch_size,
                                                                                         out_tagged_story = False, shuffle_batch = True, debug=False)
        if story_idx%20000 ==0:
            print("Negative ending(s) created for : ",story_idx, "/",total_stories)
        aug_data.append(batch_aug_stories)
        ver_aug_data.append(ver_aug_stories)

    neg_end_obj.no_samp = 0
    return aug_data, ver_aug_data


#For this function the datast needs to be pos tagged
def batches_backwards_neg_endings(neg_end_obj, endings, batch_size, contexts):

    total_stories = len(endings)
    aug_data = []
    ver_aug_data = []
    for story_idx in range(0, total_stories):

        batch_aug_stories, ver_aug_stories = neg_end_obj.backwards_words_substitution_approach(context_story = contexts[story_idx], ending_story = endings[story_idx], batch_size = batch_size)

        if story_idx%20000 ==0:
            print("Negative ending(s) created for : ",story_idx, "/",total_stories)

        aug_data.append(batch_aug_stories)
        ver_aug_data.append(ver_aug_stories)

    neg_end_obj.no_samp = 0
    return aug_data, ver_aug_data

def eliminate_tags_in_contexts(contexts_pos_tagged):


    contexts_no_tag = []
    print("CHECKS: ")
    print("LEN CONTEXTS AFTER ID ELIMINATION -> ",len(contexts_pos_tagged))
    print("LEN CONTEXTS[0] AFTER ID ELIMINATION -> ",len(contexts_pos_tagged[0]))

    for context in contexts_pos_tagged:
        context_no_tag = []
        for sentence in context:
            context_no_tag.append([word_tag[0] for word_tag in sentence])
        contexts_no_tag.append(context_no_tag)
    return contexts_no_tag


def eliminate_tags_in_val_endings(endings_pos_tagged):
    endings_no_tag = []
    for endings_batch_pos_tagged in endings_pos_tagged:
        batch_endings_no_tag = []
        for ending_pos_tagged in endings_batch_pos_tagged:

            batch_endings_no_tag.append([word_tag[0] for word_tag in ending_pos_tagged])
        endings_no_tag.append(batch_endings_no_tag)

    return endings_no_tag


def eliminate_tags_corpus(corpus_pos_tagged):
    '''
    Removes pos tagging from corpus
    :param corpus_pos_tagged:
    :return: corpus as a list of words without pos tagging, for each sentence
    '''
    corpus_no_tag = []
    for batch_pos_tagged in corpus_pos_tagged:
        batch_no_tag = []
        for sentence_pos_tagged in batch_pos_tagged:

            batch_no_tag.append([word_tag[0] for word_tag in sentence_pos_tagged])
        corpus_no_tag.append(batch_no_tag)

    return corpus_no_tag


"""UNTIL HERE, MAYBE MOVE IN PREPROCESSING IMPORTING * such that nothing has to be modified"""


"""********************************** CNN TRAIN - VALIDATION SETS GENERATOR *****************************"""


def batch_iter_val_cnn(contexts, endings, neg_end_obj, binary_verifiers, out_tagged_story = False,
                       batch_size = 2, num_epochs = 500, shuffle=True):
    """
    Generates a batch generator for the validation set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_in_contexts(contexts_pos_tagged = contexts)
        endings = eliminate_tags_in_val_endings(endings_pos_tagged = endings)

    while True:

        batches_full_stories = full_stories_together(contexts = contexts, endings = endings, list_array = True)

        total_steps = len(batches_full_stories)

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            stories_batch = batches_full_stories[batch_idx]
            binary_batch_verifier = [[int(ver), 1-int(ver)] for ver in binary_verifiers[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(binary_batch_verifier))


def batch_iter_train_cnn(contexts, endings, neg_end_obj, out_tagged_story = False,
                         batch_size = 2, num_epochs = 500, shuffle=True, test = False):
    """
    Generates a batch generator for the train set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_in_contexts(contexts_pos_tagged= contexts)
    while True:

        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end= batches_pos_neg_endings(neg_end_obj = neg_end_obj, endings = endings,
                                                              batch_size = batch_size)
        if not test:
            batches_full_stories = full_stories_together(contexts = contexts, endings = batch_endings)
        else:
            batches_full_stories = full_stories_together(contexts = contexts, endings = batch_endings)

        total_steps = len(batches_full_stories)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones

            stories_batch = batches_full_stories[batch_idx]
            verifier_batch = [[int(ver), 1-int(ver)] for ver in ver_batch_end[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(verifier_batch))


def batch_iter_backward_train_cnn(contexts, endings, neg_end_obj, out_tagged_story=False,
                                  batch_size=2, num_epochs=500, shuffle=True):
    """
    Generates a batch generator for the train set.
    """
    if not out_tagged_story:
        contexts_no_tag = eliminate_tags_in_contexts(contexts_pos_tagged=contexts)

    while True:

        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end = batches_backwards_neg_endings(neg_end_obj=neg_end_obj, endings=endings,
                                                                     batch_size=batch_size, contexts=contexts)
        batches_full_stories = full_stories_together(contexts=contexts_no_tag, endings=batch_endings,
                                                     contexts_aggregated=False)
        total_steps = len(batches_full_stories)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            # batch_size stories -> 1 positive endings + batch_size-1 negative endings ones

            stories_batch = batches_full_stories[batch_idx]
            verifier_batch = [[int(ver), 1 - int(ver)] for ver in ver_batch_end[batch_idx]]

            yield (np.asarray(stories_batch), np.asarray(verifier_batch))


"""********************************** FUNCTIONS FOR SIAMESE LSTM *****************************"""


def embedding(docs, embedding_dim):
    '''
    :param docs: array containing sentences
    :return:
    '''
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(len(encoded_docs))

    # pad documents to a max length of 4 words
    max_length = story_len
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(len(padded_docs))

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def pad_restructure_trainset(aug_data, ver_aug_data, train_context_notag):
    '''
    To put context, ending and verifier into right format for siamese LSTM
    :return: 3 arrays for context, ending (all padded to size story_len) and verifier (0 or 1)
    '''
    print("Padding train context and endings")

    # get pad index
    pad_index = get_index_from_tag(pad)

    count = 0
    story_processed = 0
    # train_structured_stories=[]
    train_structured_context = []
    train_structured_ending = []
    train_structured_verifier = []

    for story_ending in aug_data:
        for i in range(0, n_endings):

            # to have verifier as size 1
            train_structured_verifier.append(ver_aug_data[count][i])

            while len(train_context_notag[count]) < story_len:
                train_context_notag[count].append(pad_index)
            train_context_notag[count] = train_context_notag[count][0:story_len]

            train_context_notag[count] = np.asarray(train_context_notag[count])
            train_structured_context.append(train_context_notag[count])
            
            story_ending_list = story_ending[i].tolist()
            while len(story_ending_list) < story_len:
                story_ending_list.append(pad_index)
            story_ending_list = np.asarray(story_ending_list)
            train_structured_ending.append(story_ending_list)

            if story_processed % 20000 == 0:
                print("Processed ", story_processed, "/", n_endings*len(train_context_notag))
            story_processed = story_processed + 1

        count = count + 1

    train_structured_context = np.asarray(train_structured_context)
    train_structured_ending = np.asarray(train_structured_ending)
    train_structured_verifier = np.asarray(train_structured_verifier)

    # Saving datasets
    path_begin = "../data/train_begin_siameselstm"
    path_end = "../data/train_end_siameselstm"
    path_ver = "../data/train_ver_siameselstm"

    np.save(path_begin, train_structured_context)
    np.save(path_end, train_structured_ending)
    np.save(path_ver, train_structured_verifier)
    print("Saved the pos tagged corpus successfully as {}, {} and {}".format(path_begin, path_end, path_ver))

    return train_structured_context, train_structured_ending, train_structured_verifier


def pad_restructure_valset(val_context_notag, val_ending_notag, ver_val_set):
    '''
    To put context, ending and verifier into right format for siamese LSTM
    :return: 3 arrays for context, ending (all padded to size story_len) and verifier (0 or 1)
    '''
    print("Padding validation context and endings")

    # get pad index
    pad_index = get_index_from_tag(pad)

    count = 0
    story_processed = 0
    # train_structured_stories=[]
    val_structured_context = []
    val_structured_ending = []
    val_structured_verifier = []
    for story_number in range(0, len(val_context_notag)):
        for i in range(0, 2):

            # to have verifier as size 1
            while len(val_context_notag[story_number]) < story_len:
                val_context_notag[story_number].append(pad_index)
            val_context_notag[count] = val_context_notag[story_number][0:story_len]

            val_context_notag[story_number] = np.asarray(val_context_notag[count])
            val_structured_context.append(val_context_notag[story_number])

            while len(val_ending_notag[story_number][i]) < story_len:
                val_ending_notag[story_number][i].append(pad_index)

            val_ending_notag[story_number][i] = np.asarray(val_ending_notag[story_number][i])
            val_structured_ending.append(val_ending_notag[story_number][i])

            if story_processed % 20000 == 0:
                print("Processed ", story_processed, "/", 2*len(val_context_notag))
            story_processed = story_processed + 1

        count = count + 1
    
    val_structured_verifier = np.asarray(ver_val_set)
    val_structured_verifier = np.reshape(val_structured_verifier, (len(val_structured_verifier)*len(val_structured_verifier[0])))
    print("VERIFER VAL SET ",val_structured_verifier)
    val_structured_context = np.asarray(val_structured_context)

    val_structured_ending = np.asarray(val_structured_ending)


    print('val_structured_context: {}'.format(val_structured_context.shape))
    print('val_structured_ending: {}'.format(val_structured_ending.shape))
    print('val_structured_verifier: {}'.format(val_structured_verifier.shape))

    # Saving datasets
    print("Saving restructured training sets..")
    path_begin = "../data/val_begin_siameselstm"
    path_end = "../data/val_end_siameselstm"
    path_ver = "../data/val_ver_siameselstm"

    np.save(path_begin, val_structured_context)
    np.save(path_end, val_structured_ending)
    np.save(path_ver, val_structured_verifier)
    print("Saved the pos tagged corpus successfully as {}, {} and {}".format(path_begin, path_end, path_ver))

    return val_structured_context, val_structured_ending, val_structured_verifier


"""***************************CNN LSTM sentiment********************"""

def batch_iter_val_cnn_sentiment(contexts, endings, binary_verifiers, test = False, batch_size = 2):
    """
    Generates a batch generator for the validation set.
    """
    contexts = eliminate_tags_in_contexts(contexts_pos_tagged = contexts)
    endings = eliminate_tags_in_val_endings(endings_pos_tagged = endings)

    print("CHECKS:")
    print("LEN ENDINGS -> ",len(endings))
    print("LEN ENDINGS[0] -> ",len(endings[0]))
    print("LEN CONTEXTS -> ",len(contexts))
    print("LEN CONTEXTS[0] -> ",len(contexts[0]))
    print("LEN BINARY VERIFIERS -> ",len(binary_verifiers))
    print("LEN BINARY VERIFIER -> ",len(binary_verifiers[0]))

    context_sentiments = sentences_to_sentiments(contexts = contexts)
    endings_sentiments = endings_to_sentiments(endings = endings)

    while True:

        batches_full_stories = full_stories_together(contexts = context_sentiments, endings = endings_sentiments)#, list_array = True)

        total_steps = len(batches_full_stories)
        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            stories_batch = batches_full_stories[batch_idx]
            if not test:
                print(binary_verifiers[batch_idx])
                binary_batch_verifier = [[int(ver), 1-int(ver)] for ver in binary_verifiers[batch_idx]]
                print(np.asarray(stories_batch), np.asarray(binary_batch_verifier))
                yield (np.asarray(stories_batch), np.asarray(binary_batch_verifier))
            else:
                yield np.asarray(stories_batch)

