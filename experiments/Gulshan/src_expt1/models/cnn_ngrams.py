import os
import keras
from keras.layers import Embedding
from keras.models import load_model
from keras.utils import to_categorical
import sys
sys.path.append("..")
from config import *


class CNN_ngrams():
    """CNN model implementing a classifier using leaky ReLU and dropouts.
       INPUT -> sentence + pos tags as follows [(I,Sbj),(am,Verb),(bored,Adj)]
       """

    def __init__(self, train_generator, validation_generator = [], path=None,
                 batch_size = 2):
        """Initialise the model.

        If `path` is given, the model is loaded from memory instead of compiled.
        """
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.embedding_dimensions_words = 100 #as in w2v
        self.embedding_dimensions_tags = 10
        n_gram_size = 10
        #pool_stride_size = 2
        #pool_last_stride_size = 1
        stride_size = 10

        """Loading trained model for predicting"""
        if path:
            print("Loading existing model from {}".format(path))
            self.load(path)
            print("Finished loading model")
            return

        #TODO train generator + validation generator to be implemented once preprocessing is ready

        """Model creation"""
        self.model = keras.models.Sequential()

        """Embedding layer"""        

        self.model.add(Embedding(input_dim = vocabulary_size, output_dim = self.embedding_dimensions_words, input_length = story_len))

        """Blocks of layers
           First block"""
        self.model.add(keras.layers.Convolution1D(filters=25,
                                                  strides=stride_size,
                                                  kernel_size=n_gram_size,
                                                  padding="same"))
        #self.model.add(keras.layers.MaxPooling1D(pool_size=2,
        #                                         strides=1,
        #                                         padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.01))
        #self.model.add(keras.layers.Dropout(rate=0.25))

        """Second block"""
        self.model.add(keras.layers.Convolution1D(filters=40,
                                                  strides=2,
                                                  kernel_size=3,
                                                  padding="same"))

        self.model.add(keras.layers.LeakyReLU(alpha=0.01))
        #self.model.add(keras.layers.Dropout(rate=0.25))

        self.model.add(keras.layers.Flatten())

        self.model.add(keras.layers.Dense(units=2,
                                          kernel_regularizer=keras.regularizers.l2(1e-6),
                                          activation="softmax"))


        optimiser = keras.optimizers.Adam()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimiser,
                           metrics=["accuracy"])
        #print(self.model.summary())



    def train(self, save_path, steps_train = 88161, steps_val = 1871):
        """Train the model.

        Args:
            epochs (int): default: 100 - epochs to train.
            steps (int): default: len(dataset) - batches per epoch to train.
        """

        out_trained_models = '../trained_models'

        cnn_grams_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_acc",
                                                        factor=0.5,
                                                        patience=0.5,
                                                        verbose=0,
                                                        cooldown=0,
                                                        min_lr=0)
        stop_callback = keras.callbacks.EarlyStopping(monitor="val_acc",
                                                      min_delta=0.0001,
                                                      patience=11,
                                                      verbose=0,
                                                      mode="auto")


        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
                                                           write_graph=True,
                                                           write_grads=False, embeddings_freq=0,
                                                           embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(save_path, 'model.h5'),
            monitor='val_acc', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)
        
        #TODO train generator + validation generator to be implemented once preprocessing is ready
        print(self.model.summary())
        self.model.fit_generator(self.train_generator,
                                 steps_per_epoch= steps_train,
                                 verbose=2,
                                 epochs=500,
                                 shuffle = True,
                                 callbacks=[cnn_grams_callback, stop_callback, tensorboard_callback,
                                            checkpoint_callback],
                                 validation_data=self.validation_generator,
                                 validation_steps= steps_val)

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        self.model.save(path)
        print("Model saved to {}".format(path))