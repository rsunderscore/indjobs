# indjobs
Read job descriptions from indeed.  
 - tokenize
 - remove stopwords
 - stemming
 - tf/idf
 - unsupervised classification
   - clustering - KNN, Kmeans
   - hierarchical - Decision Trees
   - SVM
   - Naive Bayes
   - cosine similarity
   - logistic regression

## References
 - [Document Classification Using Python and Machine Learning](https://www.digitalvidya.com/blog/document-classification-python-machine-learning/) - Muneeb Ahmad
 - [Naive Bayes Document Classification in Python](https://towardsdatascience.com/naive-bayes-document-classification-in-python-e33ff50f937e) - Kelly Epley
 - [Machine Learning, NLP: Text Classification using scikit-learn, python and NLTK](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a) - Javed Shaikh 
   - do count vectorizer first?
   - also does SVM
  ```py  - from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```
 - [Text Classification with Python and Scikit-Learn](https://stackabuse.com/text-classification-with-python-and-scikit-learn/) - Usman Malik -- this one is more about sentiment
   - can pass stopwords to CountVectorizer
   - also uses `TfidfTransformer`
   - RandomForest

> The script above uses CountVectorizer class from the sklearn.feature_extraction.text library. There are some important parameters that are required to be passed to the constructor of the class. The first parameter is the max_features parameter, which is set to 1500. This is because when you convert words to numbers using the bag of words approach, all the unique words in all the documents are converted into features. All the documents can contain tens of thousands of unique words. But the words that have a very low frequency of occurrence are unusually not a good parameter for classifying documents. Therefore we set the max_features parameter to 1500, which means that we want to use 1500 most occurring words as features for training our classifier.
 - :star: start here  [Classification of text documents using sparse features](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) - scikit learn web site
   - more algorithm choices (15 total) incl: peceptron, Ridge, SGD, linearSVC, 3x Bayes Variants, Rocchio (nearest centroid)
   - HashingVectorizer
   - use tfidf transformer directly (without first doing count)?

Algorithm options:
- K-nearest neighbour algorithms
- SVM
- Decision trees such as ID3 or C4.5
- Naive Bayes classifier
- Expectation maximization (EM)
- Instantaneously trained neural networks
- Latent semantic indexing
- Support vector machines (SVM)
- Artificial neural network
- Concept Mining
- Rough set-based classifier
- Soft set-based classifier
- Multiple-instance learning
- Natural language processing approaches

more algorithms from [sklearn site](https://scikit-learn.org/stable/modules/clustering.html#)
- Spectral
- DBSCAN
- 

