#!/usr/bin/env python3 -W ignore::DeprecationWarning
import argparse
import sys
import datetime
import time
import negative_endings as data_aug
import tensorflow as tf
import keras
from models import cnn_ngrams, cnn_lstm_sent, SiameseLSTM, ffnn
from training_utils import *
from sentiment import *
from negative_endings import *
from preprocessing import full_sentence_story
from models.skip_thoughts import skipthoughts

# Remove tensorflow CPU instruction information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model

def _setup_argparser():
    """Sets up the argument parser and returns the arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Control program to launch all actions related to this project.")

    parser.add_argument("-m", "--model", action="store",
                        choices=["cnn_ngrams", "SiameseLSTM", "cnn_lstm", "cnn_lstm_val", "ffnn", "ffnn_val", "ffnn_val_test"],
                        default="cnn_ngrams",
                        type=str,
                        help="the model to be used, defaults to cnn_ngrams")
    parser.add_argument("-t", "--train",
                        help="train the given model",
                        action="store_true")
    parser.add_argument("-p", "--predict",
                        help="predict on a test set given the model",
                        action="store_true")
    parser.add_argument("-e", "--evaluate",
                        help="evaluate on a test set given the model",
                        action="store_true")

    args, unknown = parser.parse_known_args()

    return args


def get_latest_model():
    """Returns the latest directory of the model specified in the arguments.

    Returns:
        (path) a path to the directory.
    """
    __file__ = "run.py"
    file_path = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(file_path, "..","trained_models", args.model)):
        print("No trained model {} exists.".format(args.model))
        sys.exit(1)
    #Path to the model
    res = os.path.join(file_path, "..","trained_models", args.model)
    all_runs = [os.path.join(res, o) for o in os.listdir(res) if os.path.isdir(os.path.join(res, o))]
    res = max(all_runs, key=os.path.getmtime)
    print("Retrieving trained model from {}".format(res))
    return res


def get_submission_filename():
    """
    Returns:
        (path to directory) + filename of the submission file.
    """
    ts = int(time.time())
    submission_filename = "submission_" + str(args.model) + "_" + str(ts) + ".csv"
    submission_path_filename = os.path.join(get_latest_model(), submission_filename)

    return submission_path_filename


def initialize_negative_endings(contexts, endings):
    neg_end = data_aug.Negative_endings(contexts = contexts, endings = endings)
    neg_end.load_vocabulary()
    #Preserve here in case the vocabulary change, do not save filters and reload them
    neg_end.filter_corpus_tags()

    return neg_end

def get_verifiers_difference(Y_predict):
    total_labels = len(Y_predict)
    index = 0
    diff_verifiers =[]
    while index < total_labels:
        diff_verifiers.append(Y_predict[index] - Y_predict[index+1])
        index = index + 2
    return diff_verifiers

def get_predicted_labels(probabilities, submission_filename):
    labels = [1 if prob[0]>prob[1] else 2 for prob in probabilities]
    labels = np.asarray(labels, dtype=int)
    with open(submission_path_filename, "w+") as f:
        np.savetxt(submission_path_filename, labels.astype(int), fmt='%i', delimiter=',')

    print("Predicted endings saved in", submission_filename)
    return labels

"""********************************************** USER ACTIONS from parser ************************************************************"""

if __name__ == "__main__":

    __file__ = "run.py"
    file_path = os.path.dirname(os.path.abspath(__file__))

    args = _setup_argparser()
    out_trained_models = ""
    #Once we get ready we can decomment. This avoids creating files when things have still to be debugged
    if args.train:
        out_trained_models = os.path.normpath(os.path.join(file_path,
                                              "../trained_models/",
                                              args.model,
                                              datetime.datetime.now().strftime(r"%Y-%m-%d[%Hh%M]")))
        try:
            os.makedirs(out_trained_models)
        except OSError:
            pass
    else:
        out_trained_models = os.path.join(os.path.abspath("run.py"), "..", "trained_models", args.model)

    print("Trained model will be saved in ", out_trained_models)


    if args.train:
        """Create a field with your model (see the default one to be customized) and put the procedure to follow to train it"""
        if args.model == "cnn_ngrams":

            print("CNN grams training invoked")
            print("Loading dataset..")
            pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged(stop_words=False, lemm=True)

            print("Initializing negative endings..")
            neg_end = initialize_negative_endings(contexts = pos_train_begin_tog, endings = pos_train_end_tog)
            
            ver_val_set = generate_binary_verifiers(val_set)

            train_generator = batch_iter_backward_train_cnn(contexts = pos_train_begin_tog, endings = pos_train_end_tog, neg_end_obj = neg_end,
                                                               batch_size = 2, num_epochs = 500, shuffle=True)
            validation_generator = batch_iter_val_cnn(contexts = pos_val_begin_tog, endings = pos_val_end_tog, binary_verifiers = ver_val_set, 
                                                                  neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            #test_generator = train_utils.batch_iter_val_cnn(contexts = pos_test_begin_tog, endings = pos_test_end_tog, binary_verifiers = ver_test_set,
            #                                                      neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            #Initialize model
            #model = cnn_ngrams.CNN_ngrams(train_generator = validation_generator, validation_generator = test_generator)
            model = cnn_ngrams.CNN_ngrams(train_generator = train_generator, validation_generator = validation_generator, conceptnet=True)
            model.train(save_path = out_trained_models)

        elif args.model == "cnn_lstm":
            
            print("CNN LSTM training invoked")

            contexts_train_samp = np.load(train_set_sampled_pos_beg)
            endings_train_samp = np.load(train_set_sampled_pos_end)

            contexts_train_samp = eliminate_id(dataset = contexts_train_samp)
            endings_train_samp = eliminate_id(dataset = endings_train_samp)

            contexts_val = np.load(val_pos_begin)
            endings_val = np.load(val_pos_end)

            contexts_val = eliminate_id(dataset = contexts_val)
            endings_val = eliminate_id(dataset = endings_val)

            binary_verifiers_train = generate_binary_verifiers(dataset = train_set_sampled)
            binary_verifiers_val = generate_binary_verifiers(dataset = val_set)
            

            total_steps_train = 50000
            total_steps_val = len(contexts_val)
            print("\nTOTAL POSSIBLE TRAIN STEP ARE: ", len(contexts_train_samp))
            print("TOTAL STEPS/EPOCH FOR TRAIN CHOSEN TO BE: ",total_steps_train)
            print("TOTAL STEPS/EPOCH FOR VALIDATION ARE: ",total_steps_val,"\n")

            #Limiting the dataset
            contexts_train_samp = contexts_train_samp[0:total_steps_train]
            endings_train_samp = endings_train_samp[0:total_steps_train]
            binary_verifiers_train = binary_verifiers_train[0:total_steps_train]

            gen_train = batch_iter_val_cnn_sentiment(contexts = contexts_train_samp, endings = endings_train_samp, binary_verifiers = binary_verifiers_train)
            gen_val = batch_iter_val_cnn_sentiment(contexts = contexts_val, endings = endings_val, binary_verifiers = binary_verifiers_val)

            model = cnn_lstm_sent.Cnn_lstm_sentiment(train_generator = gen_train, validation_generator = gen_val)
            model.train(save_path = out_trained_models, steps_train = total_steps_train, steps_val = total_steps_val)

        elif args.model == "cnn_lstm_val":
            
            print("CNN LSTM val training invoked")

            contexts_val = np.load(val_pos_begin)
            endings_val = np.load(val_pos_end)

            contexts_val = eliminate_id(dataset = contexts_val)
            endings_val = eliminate_id(dataset = endings_val)


            binary_verifiers_val = generate_binary_verifiers(dataset = val_set)
            
            n_stories = len(contexts_val)
            train_indexes = np.random.choice(n_stories, int(n_stories*0.9), replace=False)

            print("Generating features array for train data...")
            X_train_begin = np.take(contexts_val, train_indexes, axis=0)
            X_train_end = np.take(endings_val, train_indexes, axis=0)
            Y_train = np.take(binary_verifiers_val, train_indexes, axis=0)

            print("Generating features array for validation data...")
            X_val_begin = np.delete(contexts_val, train_indexes, axis=0)
            X_val_end = np.delete(endings_val, train_indexes, axis=0)
            Y_val = np.delete(binary_verifiers_val, train_indexes, axis=0)
            
            total_steps_train = n_stories*0.9
            total_steps_val = n_stories*0.1

            print("TOTAL STEPS/EPOCH FOR TRAIN CHOSEN TO BE: ",total_steps_train)
            print("TOTAL STEPS/EPOCH FOR VALIDATION ARE: ",total_steps_val)

            gen_train = batch_iter_val_cnn_sentiment(contexts = X_train_begin, endings = X_train_end, binary_verifiers = Y_train)
            gen_val = batch_iter_val_cnn_sentiment(contexts = X_val_begin, endings = X_val_end, binary_verifiers = Y_val)

            model = cnn_lstm_sent.Cnn_lstm_sentiment(train_generator = gen_train, validation_generator = gen_val)
            model.train(save_path = out_trained_models, steps_train = total_steps_train, steps_val = total_steps_val)

        elif args.model == "SiameseLSTM":

            print("You chose the Siamese LSTM model")

            print('Tensorflow version: {}'.format(tf.__version__))
            print('Keras version: {}'.format(keras.__version__))  # Make sure version of keras is 2.1.4

            print("Loading dataset..")
            pos_train_begin, pos_train_end, pos_val_begin, pos_val_end = load_train_val_datasets_pos_tagged(
                together=False)

            elements = (pos_train_begin, pos_train_end, pos_val_begin, pos_val_end)

            pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged(
                together=True)

            print("Initializing negative endings..")
            neg_end = initialize_negative_endings(contexts=pos_train_begin_tog, endings=pos_train_end_tog)

            # Removing pos tags from train and validation datasets
            train_context_notag = np.asarray(eliminate_tags_corpus(pos_train_begin_tog))
            train_context_notag = train_context_notag.ravel()  # created a list

            val_context_notag = np.array(eliminate_tags_corpus(pos_val_begin_tog))
            val_context_notag = val_context_notag.ravel()
            val_ending_notag = np.array(eliminate_tags_in_val_endings((pos_val_end_tog)))

            ver_val_set = generate_binary_verifiers(val_set)

            # Creating augmented endings (wrong endings for each story in the training set)
            aug_data, ver_aug_data = batches_pos_neg_endings(neg_end_obj=neg_end,
                                                             endings=pos_train_end,
                                                             batch_size=n_endings)

            train_structured_context, train_structured_ending, train_structured_verifier = pad_restructure_trainset(
                aug_data, ver_aug_data, train_context_notag)

            val_structured_context, val_structured_ending, val_structured_verifier = pad_restructure_valset(
                val_context_notag, val_ending_notag, ver_val_set)

            model = SiameseLSTM.SiameseLSTM(seq_len=story_len,
                                            n_epoch=n_epoch,
                                            train_dataA=train_structured_context,
                                            train_dataB=train_structured_ending,
                                            train_y=train_structured_verifier,
                                            val_dataA=val_structured_context,
                                            val_dataB=val_structured_ending,
                                            val_y=val_structured_verifier,
                                            embedding_docs=full_sentence_story(train_set))

            model.train(save_path = out_trained_models)

        elif args.model == "ffnn":

            print("Loading skip-thoughts model for embedding...")
            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            print("Generating features array for train data (grab a coffee, it might take a while)...")
            X_train = ffnn.transform(train_set_sampled, encoder)
            Y_train = generate_binary_verifiers(train_set_sampled)

            print("Generating features array for validation data...")
            X_val = ffnn.transform(val_set, encoder)
            Y_val = generate_binary_verifiers(val_set)

            print("Defining batch data generators... ")
            train_generator = ffnn.batch_iter(X_train, Y_train, batch_size=64, shuffle=True)
            validation_generator = ffnn.batch_iter(X_val, Y_val, batch_size=64, shuffle=True)

            print("Initializing feed-forward neural network...")
            model = ffnn.FFNN(train_generator=train_generator, validation_generator=validation_generator)
            model.train(len(X_train), len(X_val), out_trained_models)


        elif args.model == "ffnn_val":

            print("Loading skip-thoughts model for embedding...")
            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X = ffnn.transform(val_set, encoder)
            Y = generate_binary_verifiers(val_set)

            n_stories = len(X)
            train_indexes = np.random.choice(n_stories, int(n_stories*0.9), replace=False)

            print("Generating features array for train data...")
            X_train = np.take(X, train_indexes, axis=0)
            Y_train = np.take(Y, train_indexes, axis=0)

            print("Generating features array for validation data...")
            X_val = np.delete(X, train_indexes, axis=0)
            Y_val = np.delete(Y, train_indexes, axis=0)

            print("Defining batch data generators... ")
            train_generator = ffnn.batch_iter(X_train, Y_train, batch_size=64, shuffle=True)
            validation_generator = ffnn.batch_iter(X_val, Y_val, batch_size=64, shuffle=True)

            print("Initializing feed-forward neural network...")
            model = ffnn.FFNN(train_generator=train_generator, validation_generator=validation_generator)
            model.train(len(X_train), len(X_val), out_trained_models)


        elif args.model == "ffnn_val_test":

            print("Loading skip-thoughts model for embedding...")
            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            print("Generating features array for train data...")
            X_train = ffnn.transform(val_set, encoder)
            Y_train = generate_binary_verifiers(val_set)

            print("Generating features array for validation data...")
            X_val = ffnn.transform(test_set_cloze, encoder)
            Y_val = generate_binary_verifiers(test_set_cloze)

            print("Defining batch data generators... ")
            train_generator = ffnn.batch_iter(X_train, Y_train, batch_size=64, shuffle=True)
            validation_generator = ffnn.batch_iter(X_val, Y_val, batch_size=64, shuffle=True)

            print("Initializing feed-forward neural network...")
            model = ffnn.FFNN(train_generator=train_generator, validation_generator=validation_generator)
            model.train(len(X_train), len(X_val), out_trained_models)



    if args.predict:

        """Path to the model to restore for predictions -> be sure you save the model as model.h5
           In reality, what is saved is not just the weights but the entire model structure"""
        model_path = os.path.join(get_latest_model(), "model.h5")
        """Submission file -> It will be in the same folder of the model restored to predict
           e.g trained_model/27_05_2012.../submission_modelname...."""
        submission_path_filename = get_submission_filename()

        if args.model == "cnn_ngrams":
            
            print("This prediction branch has not been implemented")

        if args.model == "SiameseLSTM":
            
            print("This prediction branch has not been implemented")

        elif args.model == "cnn_lstm" or args.model == "cnn_lstm_val":

            print("Predicting with CNN_LSTM sentiment based..")
            contexts_test = np.load(test_cloze_pos_begin)
            endings_test = np.load(test_cloze_pos_end)

            contexts_test = eliminate_id(dataset = contexts_test)
            endings_test = eliminate_id(dataset = endings_test)

            test_generator = batch_iter_val_cnn_sentiment(contexts = contexts_test, endings = endings_test, binary_verifiers = [], test = True)
            #model_class = cnn_lstm_sent.Cnn_lstm_sentiment(train_generator = [], path=model_path)
            #model = model_class.model
            model = load_model(model_path)
            Y_predict = model.predict_generator(test_generator, steps=2343)
            verifiers_differences = get_verifiers_difference(Y_predict = Y_predict)
            Y_labels = get_predicted_labels(verifiers_differences, submission_path_filename)

            print(Y_labels)
            print(Y_labels.shape)

        elif args.model == "ffnn" or args.model == "ffnn_val" or args.model == "ffnn_val_test":

            model = load_model(model_path)

            print("Loading skip-thoughts model for embedding...")

            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X_test = ffnn.transform(test_set, encoder)
            Y_predict = model.predict(X_test)
            Y_labels = get_predicted_labels(Y_predict, submission_path_filename)

    if args.evaluate:

        """Evaluate the trained model on Story Cloze test set"""

        model_path = os.path.join(get_latest_model(), "model.h5")
        submission_path_filename = get_submission_filename()

        if args.model == "ffnn" or args.model == "ffnn_val" or args.model == "ffnn_val_test":

            model = load_model(model_path)

            print("Loading skip-thoughts model for embedding...")

            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X_test = ffnn.transform(test_set_cloze, encoder)
            Y_test = generate_binary_verifiers(test_set_cloze)
            Y_test = np.asarray(Y_test)

            _, accuracy = model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
            print("[INFO] accuracy: {:.4f}%".format(accuracy * 100))
        
        elif args.model == "cnn_lstm" or args.model == "cnn_lstm_val":

            model = load_model(model_path)

            print("Loaded cnn_lstm model for evaluation..")
            
            contexts_eval = np.load(eval_pos_begin)
            endings_eval = np.load(eval_pos_end)

            contexts_eval = eliminate_id(dataset = contexts_eval)
            endings_eval = eliminate_id(dataset = endings_eval)

            binary_verifiers_eval = generate_binary_verifiers(dataset = test_set_cloze)

            gen_eval = batch_iter_val_cnn_sentiment(contexts = contexts_eval, endings = endings_eval, binary_verifiers = binary_verifiers_eval)

            _, accuracy = model.evaluate_generator(gen_eval, steps=1871)
            print("[INFO] accuracy: {:.4f}%".format(accuracy * 100))
