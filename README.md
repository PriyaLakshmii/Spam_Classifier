# Spam Classifier
A machine learning based project to classify messages as spam or not spam.

## Project Description:
This project uses a machine learning algorithm to classify messages as spam or ham(non spam).The model is trained on a dataset with labeled examples of spam and ham messages and uses natural language processing and machine learning techniques to to make predictions on new, incoming messages.

## Usage:

This project contains the following files:

##### Spam_Classification.ipynb: 
Jupyter Notebook containing the code for training and testing the spam classifier model.
##### spam.tsv: 
The dataset used for training and testing the model.
##### SVC_SPAM_CLASSIFIER.pkl: 
Trained SVC model 

To run the project, open the spam_classifier.ipynb file in Jupyter Notebook.

## Tools Used:
* Python
* pandas 
* nltk 
* scikit-learn 
* matplotlib
* seaborn

## Techniques and Algorithms:

#### The project uses the following techniques and algorithms:

#### Text pre-processing: 
The data is cleaned and prepared for modeling by removing the duplicate, punctuations and stopwords.

#### Natural language processing: 
The contents of the messages are converted into numerical representations using techniques such as TF-IDF.

#### Machine learning: 
The model is trained on the pre-processed data using algorithms such as Logistic Regression, Naive Bayes Classifier, Support Vector Classifier (SVC) and Random  Forests.

## Results:
The model achieved an accuracy of 98% on the test data. The results indicate that the model is effective in identifying spam emails with a high degree of accuracy.

## Acknowledgments:
The dataset used in this project was obtained from Kaggle.
