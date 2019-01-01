from data_utils import *
import random
import pickle
from collections import Counter
import numpy as np
from random import randint, shuffle
import data_utils as data_utils
from copy import deepcopy
import pandas as pd

class Negative_endings:
    """For reference on basic augmentation in negative endings and get inspiration from
       please see the paper An RNN-based Binary Classifier for the Story Cloze Test
       """

    def __init__(self, contexts, endings):

        print("Negative endings object created..")
        self.set_sample_probabilities()
        self.all_stories_context_pos_tagged = contexts
        self.all_stories_endings_pos_tagged = endings
        self.no_samp = 0

    def set_sample_probabilities(self):

        self.sampling_probs_tags = Counter(tags_to_sample_from)

        prob_idx = 0
        for tag in tags_to_sample_from:
            if prob_idx < len(probs_tags_to_sample_from):
                self.sampling_probs_tags[tag] = probs_tags_to_sample_from[prob_idx]
                prob_idx = prob_idx + 1

        if prob_idx < len(list(self.sampling_probs_tags)):
            print(
                "\n\nIMPORTANT WARNING: some sampling probability thresholds have not been assigned, they are left to 0\n\n")

    """******************USER FUNCTIONS: THESE FUNCTIONS ARE THE ONE TO USE FOR TRAINING*****************"""

    # Replace 5th sentence with one random of the training set
    def random_negative_ending(ending_story,  # The story can be both pos tagged and not pos tagged
                               endings,
                               contexts,
                               batch_size=2,
                               shuffle_batch=True):
        """INPUT:
                 full_training_story : matrix of 5 story sentences
                 merge_sentences : boolean, if True the output will be a unique array of story words
                                            if False the output will be a matrix of 5 story sentences
            OUTPUT:
                  training stories with the original one and others
            """

        ver_aug_stories = np.zeros(batch_size)
        ver_aug_stories[0] = 1

        for i in range(batch_size - 1):
            new_story = deepcopy(pos_tagged_story)
            new_story[-1] = deepcopy(full_train_dataset[randint(0, len(full_train_dataset) - 1)][-1]);

        if shuffle_batch:
            batch_aug_stories, ver_aug_stories = self.shuffle_story_verifier(batch_size=batch_size,
                                                                             batch_aug_stories=batch_aug_stories,
                                                                             ver_aug_stories=ver_aug_stories)
        # print(batch_aug_stories)
        # print(ver_aug_stories)

        return batch_aug_stories, ver_aug_stories

    # Replace 5th sentence with one random of the context & change words in it w.r.t to the tag
    def backwards_words_substitution_approach(self, context_story,  # The story has to be pos tagged
                                              ending_story,
                                              no_4th_context_sentence=False,
                                              out_tagged_story=False,  # Output a pos_tagged story if True
                                              batch_size=2,
                                              debug=False,
                                              # Set to True if endings and original ending context is wanted to be displayed
                                              shuffle_batch=True):
        """INPUT:
                 context_story = 4 sentences context in one matrix
                 ending_story = 1 ending (the correct one)
           OUTPUT:
                  training stories endings with the correnct one and others negative (batch_size-1)
            """

        batch_aug_endings = []
        ver_aug_stories = np.zeros(batch_size)
        ver_aug_stories[0] = 1

        if not out_tagged_story:
            batch_aug_endings.append([word_tag[0] for word_tag in ending_story[0]])
        else:
            batch_aug_endings.append(ending_story[0])

        len_ending = len(ending_story[0])
        # Create new stories with different endings and add them to the training batch
        for i in range(batch_size - 1):
            new_ending = []
            while len(new_ending) == 0:
                if no_4th_context_sentence:
                    new_ending = deepcopy(context_story[randint(0, len(context_story) - 2)]);
                else:
                    new_ending = deepcopy(context_story[randint(0, len(context_story) - 1)]);

            if debug:
                print("Original chosen context story ending before changing words: ")

                print(self.display_sentences(endings=[[word_tag[0] for word_tag in new_ending]],
                                             verifier=[], tagged_story=out_tagged_story))

            self.change_sentence(new_ending)

            padding = len_ending - len(new_ending)

            for i in range(0, padding):
                new_ending = new_ending + [(self.vocabulary[pad], self.vocabulary["."])]

            if not out_tagged_story:
                batch_aug_endings.append([word_tag[0] for word_tag in new_ending])
            else:
                batch_aug_endings.append(new_ending)

        if shuffle_batch:
            batch_aug_endings, ver_aug_stories = self.shuffle_story_verifier(batch_size=batch_size,
                                                                             batch_aug_endings=batch_aug_endings,
                                                                             ver_aug_stories=ver_aug_stories)
        if debug:
            print("All endings of the story & verifier:")
            self.display_sentences(endings=batch_aug_endings, verifier=ver_aug_stories, tagged_story=out_tagged_story)

        return batch_aug_endings, ver_aug_stories

    def words_substitution_approach(self,
                                    ending_story,  # Ending of the story
                                    out_tagged_story=False,  # Output a pos_tagged story if True
                                    batch_size=2,
                                    shuffle_batch=True,
                                    debug=True):  # If the changed endings should be displayed in charachter words

        """
        INPUT:
        ending_story: array of the ending part of the story
        batch_size: desired story wrong endings augmentation

        OUTPUT:
       
        1) 3d array batch_aug_endings -> [batch_size, len(ending_story), 2] or [batch_size, len(ending_story)]  
        2) 1d array ver_aug_stories -> [batch_size]
        
        Remark batch_aug_stories: original ending_story + (batch_size-1) ending_stories which differ AT LEAST with some grammatical component.
                                  Changed words are guaranteed to be in the vocabulary !

        Remark ver_aug_stories: contains 
                                    1 for the SINGLE correct story 
                                    all 0s for the incorrect stories
       
        After the augmentation the order of the stories in the batch is shuffled (with the corresponding verifier).

        """
        batch_aug_endings = []

        if not out_tagged_story:
            batch_aug_endings.append([word_tag[0] for word_tag in ending_story[0]])
        else:
            batch_aug_endings.append(ending_story)

        ver_aug_stories = np.zeros(batch_size)
        ver_aug_stories[0] = 1

        # print("ORIGINAL POS TAGGED STORY ENDING IS: ", pos_tagged_story[-1])

        for i in range(batch_size - 1):

            new_ending = deepcopy(ending_story)
            # print("Original ending: ", new_ending[-1])
            changed_story_ending = self.change_sentence(sentence=new_ending[-1])
            new_ending[-1] = changed_story_ending
            # print("Changed ending: ", new_ending[-1])

            if not out_tagged_story:
                batch_aug_endings.append([word_tag[0] for word_tag in new_ending[-1]])
            else:
                batch_aug_endings.append(new_ending)

        if shuffle_batch:
            batch_aug_endings, ver_aug_stories = self.shuffle_story_verifier(batch_size=batch_size,
                                                                             batch_aug_endings=batch_aug_endings,
                                                                             ver_aug_stories=ver_aug_stories)
        if debug:
            print("All endings of the story & verifier:")
            self.display_sentences(endings=batch_aug_endings, verifier=ver_aug_stories, tagged_story=out_tagged_story)

        # print(batch_aug_endings)
        # print(ver_aug_stories)

        return batch_aug_endings, ver_aug_stories


    """******************************END USER FUNCTIONS**************************"""

    def display_sentences(self, endings, verifier, tagged_story):
        for ending in endings:
            sentence = []
            vocabulary_list = list(self.vocabulary)
            # print("CHECK WHERE the tags are: ", vocabulary_list)
            if tagged_story:
                for word in ending:
                    sentence.append(vocabulary_list[word[0]])
            else:
                for word in ending:
                    sentence.append(vocabulary_list[word])
            # if sentence is not None:
            #    print(sentence)
        if len(verifier) != 0:
            print(verifier)

    def shuffle_story_verifier(self, batch_size, batch_aug_endings, ver_aug_stories):
        shuffled_idx = np.arange(batch_size)
        shuffle(shuffled_idx)
        # print(shuffled_idx)
        batch_aug_endings = np.asarray(batch_aug_endings)[shuffled_idx]
        ver_aug_stories = np.asarray(ver_aug_stories)[shuffled_idx]
        return batch_aug_endings, ver_aug_stories

    def join_story_from_sentences(self, story_sentences):

        """Join together the different sentences of the story into a unique array"""

        # NB not the best and efficient way to do that -> please change if u have more efficient algorithm
        joined_story = []
        for sentence in story_sentences:
            for word_tag in sentence:
                joined_story.append(word_tag)
            # joined_story.append(sentence)
        # print("JOINED STORY: ",joined_story)
        return joined_story

    def change_sentence_sampling_from_context(sentence, context):

        return sentence

    def change_sentence(self, sentence):

        """Check tags. If there is a noun, verb, adverb, adjective:
        1) Sample a random number from [0,1]
        2) Compare with threshold
        3) Substitute the word if sampled number > threshold with a random dataset word belonging to the same tag

        nltk.help.upenn_tagset() -> displays the different tags and meaning
        VB, VBD, VBG, VBN, VBP, VBZ grouped together as Verbs
        NN, NNP, NNPS, NNS grouped together as nouns
        PRP grouped together as pronouns
        RB, RBR, RBS grouped together as adverbs
        JJ, JJR, JJS grouped together as adjectives

        Note that each sentence has words of the type ("I", "Subj") so to substitute just the word it should be taken position 0 of the index
        
        """

        index = 0
        # max_changes = 2
        # changes  = 0
        at_least_one_change = False
        no_sampling_tags = False
        sentence_no_sampling = 0
        found_one = False

        iterations = 0

        # print("Initial setence is: ", sentence)

        while not at_least_one_change and not no_sampling_tags:
            for tagged_word in sentence:
                if tagged_word[1] in self.sampling_tags:
                    found_one = True
                    p = random.uniform(0, 1)

                    if p > self.sampling_probs_tags[tagged_word[1]]:
                        new_word = list(sentence[index])
                        new_word[0] = self.sample_from_vocab(tagged_word[1])
                        sentence[index] = tuple(new_word)
                        at_least_one_change = True

                index = index + 1
            if not found_one:
                # TODO : Decide if sampling a random word from anothe tag or if sampling
                # from all tag directly avoiding the problem
                self.no_samp = self.no_samp + 1
                no_sampling_tags = True

            index = 0
            iterations = iterations + 1

        # print("Sentence changed into: ", sentence)
        # print("Iterations needed: ", iterations)

        return sentence

    def sample_from_vocab(self, tag):

        vocab_list = self.sampling_tags[tag]

        return vocab_list[randint(0, len(vocab_list) - 1)]

    """******************GROUPING TAGS PER TYPE TO FORM SETS TO SAMPLE FROM*****************"""

    def check_for_unknown_words(self, list_of_words):

        new_list_of_words = []
        vocabulary = self.vocabulary
        nb_words = len(list_of_words)

        for i in range(0, nb_words):
            if list_of_words[i] in vocabulary:
                new_list_of_words.append(list_of_words[i])
            else:
                new_list_of_words.append(unk)

        return new_list_of_words

    def tag_and_words_vocab_to_numerical_form(self):

        print("Translating the words in sentences and tags to vocabulary indices..")
        total_vocab_idx = 0
        print("Transalting words into vocabulary indices")
        for tag in self.sampling_tags:
            word_in_tag_vocab = self.sampling_tags[tag]
            # print(self.vocabulary)

            total_vocab_idx = total_vocab_idx + len(word_in_tag_vocab)
            for word_idx in range(0, len(word_in_tag_vocab)):
                word_in_tag_vocab[word_idx] = self.vocabulary[word_in_tag_vocab[word_idx]]

            self.sampling_tags[tag] = word_in_tag_vocab

        print("Translating tags into vocabulary indices..")
        all_tags = list(self.sampling_tags)
        for tag in all_tags:
            self.sampling_tags[self.vocabulary[tag]] = self.sampling_tags.pop(tag)
        print("Done -> numerical translations")

    def define_vocab_tags(self):
        tags_to_sample_from_numerical = [self.vocabulary[tag] for tag in tags_to_sample_from]

        for tag in tags_to_sample_from_numerical:
            self.sampling_tags[tag] = list(Counter(self.sampling_tags[tag]))

        # print(self.sampling_tags)

    def sampling_tags_to_indices(self):
        all_tags = list(self.sampling_tags)
        for tag in all_tags:
            self.sampling_tags[self.vocabulary[tag]] = self.sampling_tags.pop(tag)
            self.sampling_probs_tags[self.vocabulary[tag]] = self.sampling_probs_tags.pop(tag)

    def filter_story_tags(self, tagged_story):

        """Find different tags at :https://www.nltk.org/_modules/nltk/tag/mapping.html
           nltk.help.upenn_tagset() -> displays the different tags and meaning
           See the config file to add / remove & update filtered tags fro sampling
           """

        for tagged_sent in tagged_story:

            for tagged_word in tagged_sent:

                if tagged_word[1] in self.sampling_tags:
                    self.sampling_tags[tagged_word[1]].append(tagged_word[0])

    def filter(self, corpus):

        story_number = 0

        for tagged_story in corpus:
            self.filter_story_tags(tagged_story=tagged_story)

            story_number = story_number + 1
            if story_number % 20000 == 0:
                print("Filtering stories: ", story_number, "/", len(corpus))

    def filter_corpus_tags(self):
        """Input:
           dataset
       
           Output:
           Save in different object fields (1d arrays): 
           1) All tags specified in the config (tags_to_sample_from)
 
           The arrays will be used to create an online augmentation during training
        """
        self.sampling_tags = Counter(tags_to_sample_from)

        for tag in tags_to_sample_from:
            self.sampling_tags[tag] = []

        self.sampling_tags_to_indices()

        print("Filtering contexts..")
        self.filter(corpus=self.all_stories_context_pos_tagged)
        print("Filtering endings..")
        self.filter(corpus=self.all_stories_endings_pos_tagged)

        print("Done -> filtered corpus by tags")

        self.define_vocab_tags()
        # self.tag_and_words_vocab_to_numerical_form()

    """******************FROM VOCABULARY INDICES DATASET TO CHARACTER DATASET*****************"""

    def load_vocabulary(self):

        # self.vocabulary = data_utils.load_vocabulary()

        with open(full_vocabulary_pkl, 'rb') as handle:
            self.vocabulary = pickle.load(handle)
        print("Vocabulary loaded")
        print("Vocabulary saved into negative ending object")

    def get_sentences_from_indices(self, sentence_vocab_indices):

        sentence = data_utils.get_words_from_indexes(indexes=sentence_vocab_indices, vocabulary=self.vocabulary)
        # print(sentence)

        return sentence

    def story_into_character_sentences(self, story_vocab_indices):

        story_sentences = []

        for sentence_vocab_indices in story_vocab_indices:
            story_sentences.append(self.get_sentences_from_indices(sentence_vocab_indices=sentence_vocab_indices))

        return story_sentences


def random_negative_endings(train_data):
    n_stories = len(train_data)
    context = train_data[:, :4]

    context_ravel = np.ravel(context)
    negative_endings = np.random.choice(context_ravel, n_stories, replace=False)

    stories = np.empty((n_stories, 7), dtype=object)
    stories[:, :5] = train_data
    stories[:, 5] = negative_endings

    labels = np.empty(n_stories, dtype=int)

    for i in range(n_stories):
        if random.choice([1, 2]) == 1:
            labels[i] = 1
        else:
            labels[i] = 2
            stories[i, 4], stories[i, 5] = stories[i, 5], stories[i, 4]

    stories[:, 6] = labels

    df = pd.DataFrame(stories, columns=['sen1', 'sen2', 'sen3', 'sen4', 'sen5_1', 'sen5_2', 'ans'])

    with open(train_set_sampled, "w+") as f:
        df.to_csv(train_set_sampled, sep=',', encoding='utf-8')


if __name__ == "__main__":
    sentences = load_data(train_set)
    sens = [col for col in sentences if col.startswith('sen')]
    sentences = sentences[sens].values
    random_negative_endings(sentences)
