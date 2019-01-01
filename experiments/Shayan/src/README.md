# CSE538-NLP-CommonSense
NLP Fall 2018 - Class project on ROC Stories using Common Sense

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

Over 100,000 rows of 5 sentence each to train with.

- ROCStories winter 2017 set (14 MB): https://goo.gl/0OYkPK
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


