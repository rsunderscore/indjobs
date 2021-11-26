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
  ```py  - from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```
 - another

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
