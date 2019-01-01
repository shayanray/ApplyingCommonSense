# Applying CommonSense to ROC Stories

The baseline implementation of the ROC Story Cloze Test was adapted from:
https://github.com/robertah/nlu_project_2

Common Sense word embeddings from concept net were used instead of regular glove embeddings along with various deep learning techniques:
Additional Experiments on Newsgroup20 dataset referenced from : https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

a. CNN-LSTM and CNN-BiLSTM models:

Experiments done on this model include:
* "CNN-LSTM (baseline) [15k samples]"
* "CNN-LSTM (with concept-net) [15k samples]"
* "CNN-LSTM (with concept-net + lastsentence)"
* "CNN-BiLSTM (with concept-net + lastsentence)"

* Another Experiment on Newsgroup20 dataset (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) - replacing glove embeddings by conceptnet numberbatch 


b. For FFNN(Feed Forward Neural Network) model :

* ffnn.py (init function to build the layers of the model and the train function to change the number of epochs to 5 from 50)
* training_utils.py (in the embedding function to change the file name to provide input of ConceptNet Numberbatch Embeddings)
* preprocessing.py (in the functions pos_tag_dataset and full_sentence_story to change the number of stories to 5000)

Experiments done on this model include:
* "FeedFwdNN - baseline (5k samples)"
* "FeedFwdNN + ConceptNet (5k samples)"
* "FeedFwdNN + ConceptNet Embeddings of last sentence (5k samples)"


c. For CNN-ngrams model:

* data_utils.py (to change the number of stories in the dataset)
* run.py (CNN-ngrams section of the run)
* cnn_lstm.py (init())

Experiments done on this model include:
* "CNN-ngrams (baseline) (5k samples)"
* "CNN-ngrams (with concept-net) (5k samples)"

d. Applying common sense on news domain (News-group dataset):


Some more detailed information:
# What is the task that you are addressing? (e.g., Translation, summarization, named entity recognition)
Applying common sense to the following task.
Evaluating story understanding and story generation using Story Cloze Test on ROCStories Corpora.

URL: http://cs.rochester.edu/nlp/rocstories/

# What algorithms are you implementing (or) which systems are you going to use? Give a name and a pointer (link to a paper or resource that describes the algorithms) 

To solve this problem, we intend to use common sense reasoning prediction and try the following:
1. use concept net API to enrich word embeddings and get greater semantic depth
2. Use neural networks like Bi-LSTM, REN, NPN etc (based on how much is feasible given the time)

ConceptNet paper: https://arxiv.org/pdf/1612.03975.pdf        
Machine comprehension paper: http://www.aclweb.org/anthology/S18-1119

# What dataset are you using? Do you have it in your posession? What is the size of this dataset?

Over 88,161 rows of 5 sentence each to train with.

- ROCStories spring 2016 set (13 MB): https://goo.gl/7R59b1
- Story Cloze Test spring 2016 set:
* val set: https://goo.gl/cDmS6I
* test set: https://goo.gl/iE31Qm 

# What are the main parts of your project? This depends on the type of your project. Please provide three main parts and discuss what you will do in each part. 

1. Pre-processing the sentences to feed to ConceptNet API and obtain the word embeddings. Formulate strategies to use the best conceptnet word embedding.
2. Train the network with NN architectures like Bi-LSTM, REN, NTN etc (based on available time)
3. Tabulate the  train, test accuracy observations with various methodologies, hyper parameter tuning  using ConceptNet labels embeddings.


# What is your learning objective? (This is largely for your own self. Why are you doing this project? Identify this goal first and pick a project that furthers this goal).

Aim is to help the computers understand and interact with people in a more natural way by collecting the assumptions. 

The motive of using common sense systems is that they are highly adaptive, thus bringing the computers one step closer to the processing language the way humans do. 


