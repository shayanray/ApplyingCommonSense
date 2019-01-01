import keras
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import LSTM, Embedding, Input, Merge, Dense
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np
import sys
sys.path.append("..")
from config import *
from data_utils import *
from preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy
from negative_endings import *
import pickle
from training_utils import *


'''
https://stackoverflow.com/questions/46466013/siamese-network-with-lstm-for-sentence-similarity-in-keras-gives-periodically-th
'''



def embedding(docs):
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

#
# def create_base_network(feature_dim,seq_len):
#     '''
#     :param feature_dim:
#     :param seq_len: length of sentences
#     :return:
#     '''
#
#     embeddings_index = dict()
#     f = open('../glove_data/glove.6B/glove.6B.100d.txt')
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#     print('Loaded %s word vectors.' % len(embeddings_index))
#     # create a weight matrix for words in training docs
#     embedding_matrix = zeros((vocabulary_size, 100))
#     for word, i in t.word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#
#     # define model
#     model = Sequential()
#     model.add(Embedding(inpu_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix],
#                   input_length=story_len, trainable=False))
#     # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = story_len))
#     print(model.summary())
#     model.add(LSTM(100, batch_input_shape=(None, seq_len, feature_dim),return_sequences=True))
#     # model.add(Dense(50, activation='relu'))
#     # model.add(Dense(10, activation='relu'))
#     # print(model.summary())
#     return model


def siamese(seq_len, n_epoch, train_dataA, train_dataB, train_y, val_dataA, val_dataB, val_y, embedding_docs):


    # If no base_network
    # prepare embedding
    embeddings = embedding(embedding_docs)
    print(embeddings.shape)

    # define model
    model = Sequential()
    print(model.summary())
    print("Embedding input shape: {}".format(seq_len))
    model.add(Embedding(input_dim=len(embeddings), output_dim=embedding_dim,
                        weights=[embeddings], input_shape=(seq_len,), trainable=False))
    # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = seq_len))
    print("LSTM batch input shape: {}".format([seq_len, embedding_dim]))
    model.add(LSTM(128, batch_input_shape=(None, seq_len, embedding_dim), return_sequences=False))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(1, activation='relu'))
    print(model.summary())

    input_a = Input(shape=(seq_len, ), dtype='int32')
    input_b = Input(shape=(seq_len, ), dtype='int32')

    processed_a = model(input_a)
    processed_b = model(input_b)


    # If using base_network
    # base_network = create_base_network(feature_dim, seq_len)

    # model = Sequential()
    # model.add(Embedding(inpu_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix],
    #                     input_length=story_len, trainable=False))
    # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = story_len))
    # model.add(LSTM(100, batch_input_shape=(None, seq_len, feature_dim), return_sequences=True))
    # print(model.summary())

    # input_a = Input(shape=(seq_len, ), dtype='int32')
    # input_b = Input(shape=(seq_len, ), dtype='int32')
    # processed_a = base_network(input_a)
    # processed_b = base_network(input_b)


    distance = keras.layers.Lambda(cosine_distance, output_shape=cosine_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    adam_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    print(model.summary())

    # Fitting if not generator
    # model.fit(x=[train_data[:,0], train_data[:,1]], y=[train_data[:,2]],
    #           batch_size=2,
    #           epochs=n_epoch,
    #           validation_data=([val_data[:,0], val_data[:,1]], [val_data[:,2]]))


    # Fitting if generator
    # out_trained_models = '../trained_models'
    #
    # lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
    #                                                 factor=0.5,
    #                                                 patience=0.5,
    #                                                 verbose=0,
    #                                                 cooldown=0,
    #                                                 min_lr=0)
    # stop_callback = keras.callbacks.EarlyStopping(monitor="acc",
    #                                               min_delta=0.0001,
    #                                               patience=11,
    #                                               verbose=0,
    #                                               mode="auto")
    #
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
    #                                                    write_graph=True,
    #                                                    write_grads=False, embeddings_freq=0,
    #                                                    embeddings_layer_names=None, embeddings_metadata=None)
    #
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     os.path.join(out_trained_models, 'cnn_ngrams/model.h5'),
    #     monitor='val_loss', verbose=0, save_best_only=True,
    #     save_weights_only=False, mode='auto', period=1)
    #
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=1871,
    #                     verbose=2,
    #                     epochs=n_epoch,
    #                     callbacks=[lr_callback, stop_callback, tensorboard_callback, checkpoint_callback],
    #                     validation_data=validation_generator,
    #                     validation_steps=1871)
    return model


def exponent_neg_manhattan_distance(A, B):
    """
    Helper function used to estimate similarity between the LSTM outputs
    :param X:  output of LSTM with input X
    :param Y:  output of LSTM with input Y
    :return: Manhattan distance between input vectors
    """
    return K.exp(-K.sum(K.abs(A - B), axis=1, keepdims=True))


def cosine_distance(vecs):
    #I'm not sure about this function too
    y_true, y_pred = vecs
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def cosine_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print((shape1[0], 1))
    return (shape1[0], 1)


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
        #print(batch_endings_no_tag)
        corpus_no_tag.append(batch_no_tag)

    return corpus_no_tag

def initialize_negative_endings(contexts, endings):
    neg_end = Negative_endings(contexts = contexts, endings = endings)
    neg_end.load_vocabulary()
    #Preserve here in case the vocabulary change, do not save filters and reload them
    neg_end.filter_corpus_tags()

    return neg_end


if __name__ == '__main__':
    import tensorflow as tf
    import keras

    print(tf.__version__)
    print(keras.__version__) # Make sure version of keras is 2.1.4



    print("Initializing negative endings...")
    pos_train_begin, pos_train_end, pos_val_begin, pos_val_end = load_train_val_datasets_pos_tagged(together=False)

    # For Testing, reducing training set to 5 stories
    sample_size = len(pos_train_begin)
    print('sample size: {}'.format(sample_size))
    sample_size=5

    pos_train_begin = pos_train_begin[:sample_size]
    pos_train_end = pos_train_end[:sample_size]
    pos_val_begin = pos_val_begin
    pos_val_end = pos_val_end
    pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged(together=True)
    pos_train_end_tog = pos_train_end_tog[:sample_size]
    pos_train_begin_tog = pos_train_begin_tog[:sample_size]
    pos_val_begin_tog = pos_val_begin_tog
    pos_val_end_tog = pos_val_end_tog

    neg_end = initialize_negative_endings(contexts=pos_train_begin_tog, endings=pos_train_end_tog)

    ver_val_set = generate_binary_verifiers(val_set)

    train_context_notag = np.asarray(eliminate_tags_corpus(pos_train_begin_tog))
    train_context_notag = train_context_notag.ravel() #created a list

    val_context_notag = np.array(eliminate_tags_corpus(pos_val_begin_tog))
    val_context_notag = val_context_notag.ravel()

    val_ending_notag = np.array(eliminate_tags_in_val_endings((pos_val_end_tog)))

    n_endings=3
    aug_data, ver_aug_data = batches_pos_neg_endings(neg_end_obj=neg_end,
                                                     endings=pos_train_end,
                                                     batch_size=n_endings)

    train_structured_context, train_structured_ending, train_structured_verifier = pad_restructure_trainset(aug_data, ver_aug_data, train_context_notag)
    val_structured_context, val_structured_ending, val_structured_verifier = pad_restructure_valset(val_context_notag, val_ending_notag, ver_val_set)

    #
    n_epoch = 1
    epochs=n_epoch

    model = siamese(seq_len=story_len,
                    n_epoch = n_epoch,
                    train_dataA=train_structured_context,
                    train_dataB=train_structured_ending,
                    train_y=train_structured_verifier,
                    val_dataA= val_structured_context,
                    val_dataB=val_structured_ending,
                    val_y=val_structured_verifier,
                    embedding_docs=full_sentence_story(train_set))

    train_dataA = train_structured_context
    train_dataB = train_structured_ending
    train_y = train_structured_verifier
    val_dataA = val_structured_context
    val_dataB = val_structured_ending
    val_y = val_structured_verifier

    model.fit(x=[train_dataA, train_dataB], y=train_y,
              batch_size=1,
              epochs=n_epoch,
              validation_data=([val_dataA, val_dataB], val_y))

